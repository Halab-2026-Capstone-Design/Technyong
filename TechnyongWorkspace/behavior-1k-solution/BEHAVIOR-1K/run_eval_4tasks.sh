#!/bin/bash
set -e

REPO=/home/data/Technyong_workspace/behavior-1k-solution/BEHAVIOR-1K
PYTHON=/root/miniconda3/envs/behavior/bin/python3
EVAL_SCRIPT=${REPO}/OmniGibson/omnigibson/learning/eval.py

export PYTHONPATH="${REPO}/joylo:${REPO}/OmniGibson:${PYTHONPATH}"
export MPLCONFIGDIR=/tmp/matplotlib-behavior
export OMNIGIBSON_HEADLESS=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export OMNIGIBSON_DATA_PATH=/home/data/Technyong_workspace/BEHAVIOR-1K/datasets

HOST=127.0.0.1
PORT=8000
INSTANCE_IDS="[0,1,2,3,4,5,6,7,8,9]"

TASKS=(
    "turning_on_radio:task0"
    "putting_away_Halloween_decorations:task2"
    "can_meat:task4"
    "make_microwave_popcorn:task40"
)

echo "=== Policy server 연결 확인 ==="
curl -s --max-time 3 http://${HOST}:${PORT}/healthz && echo " OK" || { echo "FAIL - policy server 확인 필요"; exit 1; }
echo ""

for entry in "${TASKS[@]}"; do
    TASK_NAME="${entry%%:*}"
    TASK_LABEL="${entry##*:}"
    LOG_PATH="${REPO}/eval_logs/${TASK_LABEL}"

    echo "======================================"
    echo "▶ Task: ${TASK_NAME} (${TASK_LABEL})"
    echo "  Log : ${LOG_PATH}"
    echo "======================================"

    ${PYTHON} ${EVAL_SCRIPT} \
        policy=websocket \
        model.host=${HOST} \
        model.port=${PORT} \
        task.name=${TASK_NAME} \
        eval_instance_ids=${INSTANCE_IDS} \
        log_path=${LOG_PATH}

    echo "✓ ${TASK_NAME} 완료"
    echo ""
done

echo "======================================"
echo "전체 평가 완료!"
echo "결과:"
for entry in "${TASKS[@]}"; do
    TASK_LABEL="${entry##*:}"
    echo "  ${REPO}/eval_logs/${TASK_LABEL}/"
done
