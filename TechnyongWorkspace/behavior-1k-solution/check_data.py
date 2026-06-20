import pandas as pd
import numpy as np

try:
    # 우리가 아까 확인한 파일 경로
    f = '/root/data/behavior_224_rgb/data/task-0000/episode_00000010.parquet'
    df = pd.read_parquet(f)
    row = df.iloc[0].to_dict()
    
    # 평면적인 컬럼명을 딕셔너리 구조로 변환 테스트
    new_row = {}
    for col, val in row.items():
        if '.' in col:
            parts = col.split('.')
            d = new_row
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val
        else:
            new_row[col] = val
            
    print("\n" + "="*40)
    print("✅ 데이터 구조 분석 결과")
    print("-" * 40)
    print(f"1. 최상위 항목: {list(new_row.keys())}")
    
    if 'observation' in new_row:
        print(f"2. 'observation' 내부 항목: {list(new_row['observation'].keys())}")
    
    # 이미지 데이터가 숨어있는지 확인
    obs_keys = str(new_row.get('observation', {})).lower()
    if 'image' in obs_keys or 'rgb' in obs_keys:
        print("3. 이미지 데이터: 발견됨 ✨")
    else:
        print("3. 이미지 데이터: 없음 ❌ (다른 곳에 있을 가능성)")
    print("="*40 + "\n")

except Exception as e:
    print(f"❌ 분석 실패: {e}")
