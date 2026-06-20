import os

file_path = 'scripts/compute_norm_stats.py'
with open(file_path, 'r') as f:
    content = f.read()

# 데이터를 읽은 직후 구조를 변환하는 로직 주입
target = "            episode = pd.read_parquet(f)"
patch = """            episode_df = pd.read_parquet(f)
            episode = []
            for _, row in episode_df.iterrows():
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
                episode.append(new_row)"""

if target in content:
    new_content = content.replace(target, patch)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✅ scripts/compute_norm_stats.py 패치 완료!")
else:
    print("❌ 대상 코드를 찾지 못했습니다. 이미 수정되었거나 형식이 다릅니다.")
