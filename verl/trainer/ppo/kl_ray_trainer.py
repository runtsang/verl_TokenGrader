
import logging
import torch
import torch.distributed
import ray
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.utils.debug import marked_timer
from verl import DataProto
from verl.trainer.ppo.kl_core_algos import apply_selective_kl_anchoring
from verl.workers.kl_fsdp_workers import KLActorRolloutRefWorker
from verl.trainer.ppo.ray_trainer import compute_reward, compute_reward_async, compute_advantage, apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.metric_utils import reduce_metrics

logger = logging.getLogger(__name__)

class KLRayPPOTrainer(RayPPOTrainer):

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        # Use KLActorRolloutRefWorker
        actor_rollout_cls = KLActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, ray_worker_group_cls

    def fit(self):
        # Copy fit loop but replace `apply_kl_penalty` with `apply_selective_kl_anchoring`
        
        # Imports needed inside fit
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from tqdm import tqdm
        from pprint import pprint
        import numpy as np
        import uuid
        from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
        from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            self._validate()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        
        # Reward Shaping / KL Params
        rs_config = self.config.algorithm.get("reward_shaping", {})
        window_size = rs_config.get("window_size", 10)
        min_windows = rs_config.get("min_windows", 3)
        max_windows = rs_config.get("max_windows", 10)
        sparsity_coeff = rs_config.get("sparsity_coeff", 0.0) # gamma
        initial_scale = rs_config.get("initial_scale", 2.0)
        final_scale = rs_config.get("final_scale", 1.25)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # Start Profile omitted...
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        # CALL CUSTOM GENERATE
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences_with_density(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output) # Warning: Might lack density if async

                    # Merge
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Reward
                    with marked_timer("reward", timing_raw, color="yellow"):
                         if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer)
                         else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Log Probs (Old)
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    
                    # Ref Log Probs (Required for KL)
                    if self.use_reference_policy:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                    # Advantage & KL
                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        # === APPLY SELECTIVE KL ANCHORING ===
                        if self.config.algorithm.use_kl_in_reward:
                            if "density_scores" in batch.batch.keys():
                                batch, kl_metrics = apply_selective_kl_anchoring(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    window_size=window_size,
                                    min_windows=min_windows,
                                    max_windows=max_windows,
                                    sparsity_coeff=sparsity_coeff,
                                    initial_scale=initial_scale,
                                    final_scale=final_scale,
                                    global_step=self.global_steps,
                                    total_steps=self.total_training_steps
                                )
                                metrics.update(kl_metrics)
                            else:
                                # Fallback if density missing
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute Advantages (Standard GRPO)
                        # Now that `token_level_rewards` are modified with selective KL, 
                        # `compute_advantage` will just calculate GAE or GRPO on them.
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm
                        )

                    # Update Actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    
                    # Logging ...
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1
                    
                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        self._save_checkpoint()
        
        progress_bar.close()
