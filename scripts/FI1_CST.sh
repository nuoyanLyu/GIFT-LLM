set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

LOCAL_PATH="/data1/lvnuoyan/llm_model/gift/mn_batch"
LOG_DIR="/home/lvnuoyan/RAGEN/logs/mn_batch"
MODEL_PATH="/data1/lvnuoyan/llm_model/Qwen2.5-1.5B-Instruct"
mkdir -p "$LOG_DIR" # 如果目录不存在，则创建它

# 获取当前时间，格式为 YYYY-MM-DD-HHMMSS
TIMESTAMP=$(date +"%m-%d-%H-%M")

WANDB_MODE=offline RAY_DEDUP_LOGS=0 python train.py --config-name _25_mn_batch \
 system.CUDA_VISIBLE_DEVICES=\"3\" \
 model_path=$MODEL_PATH \
 trainer.default_local_dir=$LOCAL_PATH \
 trainer.total_training_steps=300 \
 trainer.n_gpus_per_node=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 trainer.experiment_name=mn-batch2 \
 $USE_GRPO 2>&1 | tee "$LOG_DIR/grpo_${TIMESTAMP}.log" 
