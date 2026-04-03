#!/bin/bash

# 1. B1K 공식 태스크 중 우선 학습할 실제 10개 태스크 리스트
TASKS=(
    "picking_up_trash"             # 쓰레기 줍기
    "collecting_aluminum_cans"     # 알루미늄 캔 수거하기
    "storing_groceries"            # 식료품 보관하기
    "cleaning_shoes"               # 신발 닦기
    "putting_away_toys"            # 장난감 정리하기
    "throwing_away_leftovers"      # 남은 음식 버리기
    "organizing_cleaning_supplies" # 청소 용품 정리하기
    "setting_the_table"            # 식탁 차리기
    "packing_a_box"                # 상자 포장하기
    "washing_dishes"               # 설거지하기
)

echo "🚀 [Step 2] 10개 태스크 우선 학습을 시작합니다!"

# 2. 리스트에 있는 태스크를 하나씩 꺼내서 10번 반복
for TASK in "${TASKS[@]}"; do
    echo "========================================="
    echo "▶️ 현재 학습 태스크: $TASK"
    echo "========================================="
    
    python scripts/run_rft_pipeline.py \
        --task_name $TASK \
        --sft_checkpoint outputs/checkpoints/best_base_model \
        --rft_rounds 3 \
        --rollout_instances 10
        
    echo "✅ [$TASK] 학습 완료! 메모리를 정리합니다."
    sleep 5 
done

echo "🎉 10개 태스크 우선 학습이 모두 안전하게 종료되었습니다!"
