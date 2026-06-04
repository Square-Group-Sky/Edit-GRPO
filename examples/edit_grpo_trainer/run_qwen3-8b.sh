# Edit-GRPO: GRPO with gradient cosine similarity for query editing.
#
# Key differences from standard GRPO:
#   - Advantage normalization uses mean/std from non-q1 samples only (q1 excluded)
#   - Gradient cosine similarity cos(g_i, g_q1) is added to non-q1 advantages
#   - Reward is format_score + accuracy_score (rule-based combined reward)
#
# Requirements:
#   - use_fused_kernels=False (default) — logits must be materialized
#   - use_remove_padding=False — MVP does not support rmpad path
#   - use_dynamic_bsz=False (default) — groups must stay intact in micro-batches
#
# Data format (standard mode — same prompt, multiple responses):
#   - Standard verl data format: each sample has one prompt
#   - rollout.n=G: generates G responses per prompt
#   - First response in each group = q1 (target query)
#   - Remaining responses = generalization
#
# Data format (pre-grouped mode — different prompts per group, set rollout.n=1):
#   - Each sample has its own prompt, pre-assigned uid groups samples
#   - Data must include "uid" in non_tensor_batch to preserve grouping
#   - Optionally include "is_q1" bool array for explicit q1 marking
#
# Tested with: Qwen3-8B on GSM8K-style math reasoning tasks.

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_gradient_cos \
    algorithm.grpo_gradient_cos.enable=True \
    algorithm.grpo_gradient_cos.lambda_cos=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_edit_grpo' \
    trainer.experiment_name='qwen3_8b_edit_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
