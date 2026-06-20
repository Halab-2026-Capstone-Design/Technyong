#!/bin/bash
# 5개 태스크 순차 eval 스크립트
# 태스크: turning_on_radio, picking_up_trash, bringing_water, tidying_bedroom, putting_shoes_on_rack
# 인스턴스: 0~9

BASE=/home/data/Technyong_workspace/behavior-1k-solution/BEHAVIOR-1K

export PYTHONPATH="$BASE/joylo:$BASE/OmniGibson:$PYTHONPATH"
export MPLCONFIGDIR=/tmp/matplotlib-behavior
export OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export OMNIGIBSON_DATA_PATH=/home/data/Technyong_workspace/BEHAVIOR-1K/datasets

TASKS=(
    "turning_on_radio"
    "picking_up_trash"
    "bringing_water"
    "tidying_bedroom"
    "putting_shoes_on_rack"
)

cd $BASE

for TASK in "${TASKS[@]}"; do
    LOG_DIR="./eval_logs/${TASK}"
    LOG_FILE="/tmp/eval_${TASK}.log"

    echo ""
    echo "=================================================="
    echo "  Starting: $TASK"
    echo "  Log: $LOG_FILE"
    echo "  Time: $(date)"
    echo "=================================================="

    python OmniGibson/omnigibson/learning/eval.py \
        policy=websocket \
        model.host=127.0.0.1 \
        model.port=8000 \
        task.name=$TASK \
        "eval_instance_ids=[0,1,2,3,4,5,6,7,8,9]" \
        log_path=$LOG_DIR 2>&1 | tee $LOG_FILE

    EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    echo "  Finished: $TASK (exit code: $EXIT_CODE) at $(date)"

    # 태스크 간 10초 대기 (시뮬레이터 정리 시간)
    sleep 10
done

echo ""
echo "=================================================="
echo "  All 5 tasks completed!"
echo "  Results in: $BASE/eval_logs/"
echo "=================================================="
