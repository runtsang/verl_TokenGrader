
import hydra
from verl.trainer.ppo.rs_ray_trainer import RSRayPPOTrainer
from verl.trainer.main_ppo import run_ppo

@hydra.main(config_path="verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Inject Reward Shaping specific configs if passed via command line overrides
    # Hydra handles this automatically if keys match.
    # But new keys need to be allowed or struct disabled.
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False) # Allow new keys
    
    # Run PPO with Custom Trainer
    # We use run_ppo but pass our custom task_runner logic? 
    # run_ppo in main_ppo.py instantiates TaskRunner then runs it.
    # RayPPOTrainer IS the class that runs the loop. 
    # run_ppo actually calls `RayPPOTrainer`? 
    # No, `run_ppo` in `main_ppo.py`:
    # 1. ray.init
    # 2. Instantiate `TaskRunner` (remote).
    # 3. Call `runner.run.remote(config)`.
    # 4. `TaskRunner.run` instantiates `RayPPOTrainer`.
    
    # So we need to override `TaskRunner` or `RayPPOTrainer` instantiation inside TaskRunner.
    # `TaskRunner` in `main_ppo.py` hardcodes `RayPPOTrainer`.
    
    # We need to Monkey Patch or subclass TaskRunner and pass it to run_ppo.
    
    from verl.trainer.main_ppo import TaskRunner
    
    class RSTaskRunner(TaskRunner):
        def add_actor_rollout_worker(self, config):
            """Override to use RSActorRolloutRefWorker instead of default ActorRolloutRefWorker"""
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.rs_fsdp_workers import RSActorRolloutRefWorker
            from verl.trainer.ppo.ray_trainer import Role
            
            # Use our custom worker class that has generate_sequences_with_density
            actor_rollout_cls = RSActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
            
            return actor_rollout_cls, ray_worker_group_cls
        
        def run(self, config):
            # This is largely copied from `TaskRunner.run` but uses `RSRayPPOTrainer`
            # Actually `TaskRunner.run` calls `RayPPOTrainer(...)`.
            
            # We can override `__init__` or `run`.
            # Let's override `run`.
            
            # COPY-PASTE imports from main_ppo
            import os
            import socket
            from pprint import pprint
            from omegaconf import OmegaConf
            from verl.utils.fs import copy_to_local
            from verl.trainer.ppo.reward import load_reward_manager
            from verl.utils.config import validate_config
            from verl.trainer.ppo.utils import need_critic, need_reference_policy
            from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
            from verl.utils.dataset.rl_dataset import collate_fn
            from verl.utils import hf_processor, hf_tokenizer # imports inside run
            
            print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)

            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_reward_model_worker(config)
            self.add_ref_policy_worker(config, actor_rollout_cls)

            validate_config(
                config=config,
                use_reference_policy=need_reference_policy(self.role_worker_mapping),
                use_critic=need_critic(config),
            )

            local_path = copy_to_local(
                config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
            )
            
            trust_remote_code = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
            )

            class FallbackRewardManager:
                def __init__(self, valid_reward_fn):
                    self.valid_reward_fn = valid_reward_fn
                
                def __call__(self, data, return_dict=False):
                    if "data_source" in data.non_tensor_batch:
                        ds = data.non_tensor_batch["data_source"]
                        # Replace empty strings or None with 'aime'
                        for i in range(len(ds)):
                            if not ds[i]: # Empty string or None
                                ds[i] = 'aime'
                    return self.valid_reward_fn(data, return_dict)
            
            val_reward_fn = FallbackRewardManager(val_reward_fn)

            resource_pool_manager = self.init_resource_pool_mgr(config)

            train_dataset = create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor, is_train=True,
                max_samples=config.data.get("train_max_samples", -1),
            )
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )
            train_sampler = create_rl_sampler(config.data, train_dataset)

            # === USE CUSTOM TRAINER ===
            trainer = RSRayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )
            trainer.init_workers()
            trainer.fit()

    # Pass custom task runner to run_ppo
    import ray
    # We need to wrap it in ray.remote if run_ppo expects it
    # run_ppo does: `task_runner_class = ray.remote(num_cpus=1)(TaskRunner)`
    
    # We pass the class itself? 
    # run_ppo signature: `def run_ppo(config, task_runner_class=None)`
    
    # So we pass the UNDECORATED class?
    # No, run_ppo checks `if task_runner_class is None: task_runner_class = ray.remote...`
    # So we should pass the decorated one.
    
    remote_runner = ray.remote(num_cpus=1)(RSTaskRunner)
    run_ppo(config, task_runner_class=remote_runner)

if __name__ == "__main__":
    main()
