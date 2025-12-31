#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"
fi

# Execute kl_main_kl.py
python3 kl_main_kl.py \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/train.parquet \
    data.val_files=$HOME/deepscaler/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name='rs_grpo' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}" \
    +reward_config.sigmoid_reward=False \
    ++custom_reward_function.path=$PWD/kl_reward_utils.py \
    ++custom_reward_function.name=kl_compute_score \
    +reward_config.linear_reward=True \
    +reward_config.multiplier_reward=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +reward_config.alpha=0.0003 \
    +algorithm.reward_shaping.window_size=10 \
    +algorithm.reward_shaping.min_windows=3 \
    +algorithm.reward_shaping.max_windows=10 \
    +algorithm.reward_shaping.initial_scale=2.0 \
    +algorithm.reward_shaping.final_scale=1.25 \
    +algorithm.reward_shaping.sparsity_coeff=0.1 