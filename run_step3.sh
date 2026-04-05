#!/bin/bash

# ==========================================
# [그룹 1: 정리 정돈 및 물건 배치]
# ==========================================
GROUP_1=(
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

# ==========================================
# [그룹 2: 청소 및 집안 가꾸기]
# ==========================================
GROUP_2=(
    "wiping_the_table"             # 식탁 닦기
    "vacuuming_the_floor"          # 진공청소기 돌리기
    "sweeping_the_floor"           # 바닥 쓸기
    "mopping_the_floor"            # 바닥 걸레질하기
    "dusting_shelves"              # 선반 먼지 털기
    "cleaning_windows"             # 창문 닦기
    "making_the_bed"               # 침대 정리하기
    "watering_plants"              # 화분에 물 주기
    "taking_out_the_trash"         # 쓰레기 내다 버리기
    "sorting_mail"                 # 우편물 분류하기
)

# ==========================================
# [그룹 3: 주방 및 가전제품 조작]
# ==========================================
GROUP_3=(
    "turning_on_radio"             # 라디오 켜기
    "opening_the_fridge"           # 냉장고 열기 (정리)
    "cleaning_the_microwave"       # 전자레인지 청소하기
    "cleaning_the_oven"            # 오븐 청소하기
    "loading_the_dishwasher"       # 식기세척기에 그릇 넣기
    "unloading_the_dishwasher"     # 식기세척기에서 그릇 빼기
    "stocking_the_pantry"          # 식료품 창고 채우기
    "boiling_water"                # 물 끓이기
    "preparing_a_salad"            # 샐러드 준비하기
    "making_a_sandwich"            # 샌드위치 만들기
)

# ==========================================
# [그룹 4: 세탁 및 옷장 정리]
# ==========================================
GROUP_4=(
    "washing_clothes"              # 세탁기 돌리기
    "drying_clothes"               # 건조기 돌리기
    "folding_laundry"              # 빨래 개기
    "ironing_clothes"              # 옷 다림질하기
    "organizing_a_closet"          # 옷장 정리하기
    "packing_a_suitcase"           # 여행 가방 싸기
    "unpacking_a_suitcase"         # 여행 가방 풀기
    "arranging_books"              # 책장에 책 꽂기
    "cleaning_the_bathroom"        # 화장실 청소하기
    "cleaning_the_kitchen"         # 주방 청소하기
)

# ==========================================
# [그룹 5: 도구 사용 및 특수 작업]
# ==========================================
GROUP_5=(
    "chopping_an_onion"            # 양파 썰기
    "setting_up_a_coffee_station"  # 커피 스테이션 세팅하기
    "feeding_a_pet"                # 반려동물 먹이 주기
    "cleaning_a_litter_box"        # 배변 훈련함(고양이 화장실) 치우기
    "replacing_a_light_bulb"       # 전구 교체하기
    "assembling_furniture"         # 가구 조립하기
    "mowing_the_lawn"              # 잔디 깎기
    "washing_a_car"                # 세차하기
    "cleaning_a_bicycle"           # 자전거 닦기
    "setting_up_a_tent"            # 텐트 설치하기
)

# 모든 그룹을 하나의 배열로 병합
ALL_TASKS=("${GROUP_1[@]}" "${GROUP_2[@]}" "${GROUP_3[@]}" "${GROUP_4[@]}" "${GROUP_5[@]}")

echo "🚀 [Step 3] 50개 태스크 본격 학습을 시작합니다! (2개씩 병렬 실행)"

# 배열의 인덱스를 이용해 반복문 실행
for i in "${!ALL_TASKS[@]}"; do
    TASK="${ALL_TASKS[$i]}"
    
    echo "▶️ 실행 시작: $TASK"
    
    # 백그라운드(&)로 실행하며, 태스크별로 로그 파일 분리
    python scripts/run_rft_pipeline.py \
        --task_name $TASK \
        --sft_checkpoint outputs/checkpoints/best_base_model \
        --rft_rounds 3 \
        --rollout_instances 10 > "log_${TASK}.txt" 2>&1 &
        
    # 2개가 실행될 때마다 대기 (짝수 번째 인덱스 도달 시)
    if (((i + 1) % 2 == 0)); then
        echo "⏳ 2개 태스크가 모두 끝날 때까지 대기합니다..."
        wait
        echo "✅ 1개 그룹(2개 태스크) 완료! 메모리 정리 후 다음 그룹으로 넘어갑니다."
        sleep 5
    fi
done

# 혹시 남아있을 수 있는 백그라운드 작업을 위해 마지막으로 한 번 더 대기
wait
echo "🎉 50개 태스크 RFT 최종 학습이 모두 완료되었습니다! (로그 파일을 확인하세요)"
