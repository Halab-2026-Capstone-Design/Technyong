from huggingface_hub import hf_hub_download
import os

repo = "IliaLarchenko/behavior_224_rgb"
local_dir = "/home/data/Technyong_workspace/behavior-1k-solution/data/IliaLarchenko/behavior_224_rgb"

# 핵심 설명서 파일들 리스트
meta_files = [
    "meta/episodes.jsonl",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.jsonl",
    "dataset_info.json"
]

for f in meta_files:
    try:
        print(f"📥 Downloading {f}...")
        hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=local_dir)
    except Exception as e:
        print(f"❌ Failed {f}: {e}")
