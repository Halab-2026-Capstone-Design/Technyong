import os
from huggingface_hub import snapshot_download

local_dir = "./data/behavior_224_rgb"

# 폴더 존재 여부와 상관없이 다운로드 로직으로 바로 진입합니다.
try:
    snapshot_download(
        repo_id="IliaLarchenko/behavior_224_rgb",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        max_workers=1,
        resume_download=True
    )
    print("Success: Dataset download completed!")
except Exception as e:
    print(f"Error: {e}")
