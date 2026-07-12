#!/usr/bin/env bash
set -x


export PYTHONUNBUFFERED=1

ROLLOUT_NUMBER=${ROLLOUT_NUMBER:-8}
COUNTERFACTUAL_K=${COUNTERFACTUAL_K:-4}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-20}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-data/MultiTask/multitask_train.parquet}
VAL_FILE=${VAL_FILE:-data/MultiTask/multitask_test.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/think_tools_multi}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_${ROLLOUT_NUMBER}}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${OUTPUT_DIR}/${EXPERIMENT_NAME}}

mkdir -p "${CHECKPOINT_PATH}"

python3 -m verl.trainer.main \
    config=./examples/config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.system_prompt=./examples/format_prompt/system.jinja \
    data.user_prompt=./examples/format_prompt/user.jinja \
    data.user_direct_prompt=./examples/format_prompt/user_direct.jinja \
    data.user_think_prompt=./examples/format_prompt/user_think.jinja \
    data.max_prompt_length=8192 \
    data.mode=auto \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=${ROLLOUT_NUMBER} \
    worker.rollout.mode=auto \
    worker.rollout.counterfactual_k=${COUNTERFACTUAL_K} \
    worker.rollout.counterfactual_temperature=0.8 \
    worker.rollout.counterfactual_top_p=0.95 \
    worker.reward.reward_function=./examples/reward_function/auto_control.py:compute_score \
    worker.reward.reward_function_kwargs.must_think_weight=0.03 \
    worker.reward.reward_function_kwargs.safe_direct_weight=0.03 \
    worker.reward.reward_function_kwargs.must_think_margin=0.5 \
    worker.reward.reward_function_kwargs.safe_direct_min_success=0.75 \
    worker.reward.reward_function_kwargs.safe_direct_max_gain=-0.25 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${TRAIN_EPOCHS} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path="${CHECKPOINT_PATH}"
