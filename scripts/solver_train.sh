set -x
solver_model_path=$1
questioner_model_path=$2
experiment_name=$3

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
python question_generate/question_generator.py \
    --model_path $questioner_model_path \
    --input_file /root/autodl-tmp/data/geo3k/images \
    --output_file_path results/${experiment_name}/questions.json

sleep 5
echo 'start evaluate generated question'
python question_evaluate/question_evaluator.py \
    --model_path $solver_model_path \
    --question_file results/${experiment_name}/questions.json \
    --output_file_path results/${experiment_name}/answers.json
sleep 5
echo 'Packaging the JSON file into parquet format'
python examples/data_preprocess/geo3k_solver.py \
    --json_path results/${experiment_name}/answers.json \
    --local_save_dir results/${experiment_name}/
sleep 1
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=results/${experiment_name}/train.parquet \
    data.val_files=data/geo3k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$solver_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4\
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["swanlab"]' \
    trainer.project_name='self_train_geo3k' \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 

echo 'merging model'
PROJECT_NAME='self_train_geo3k'
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${experiment_name}"

LATEST_STEP=$(ls -1 $CHECKPOINT_DIR | grep "global_step_" | sort -V | tail -1)

if [ -n "$LATEST_STEP" ]; then
    echo "Found latest checkpoint: $LATEST_STEP"
    
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${CHECKPOINT_DIR}/${LATEST_STEP}/actor \
        --target_dir ${CHECKPOINT_DIR}/${LATEST_STEP}/actor/huggingface
    
    echo "Model merged successfully: ${CHECKPOINT_DIR}/${LATEST_STEP}/actor/huggingface"
else
    echo "Warning: No checkpoint found in $CHECKPOINT_DIR"
fi
