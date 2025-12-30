
import hydra
from verl.trainer.ppo.kl_ray_trainer import KLRayPPOTrainer
from verl.trainer.main_ppo import run_ppo

@hydra.main(config_path="verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    
    from verl.trainer.main_ppo import TaskRunner
    
    class KLTaskRunner(TaskRunner):
        def run(self, config):
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
            from verl.utils import hf_processor, hf_tokenizer

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

            trainer = KLRayPPOTrainer(
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

    import ray
    remote_runner = ray.remote(num_cpus=1)(KLTaskRunner)
    run_ppo(config, task_runner_class=remote_runner)

if __name__ == "__main__":
    main()
