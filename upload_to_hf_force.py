import os
import sys
from huggingface_hub import HfApi

def main():
    repo_id = "kabiriitb/ppo_advantage_estimator"
    api = HfApi()
    
    # The folders we want to force upload
    target_folders = ["models", "figures", "results"]
    
    for folder in target_folders:
        if not os.path.exists(folder):
            print(f"Skipping {folder}, does not exist.")
            continue
            
        print(f"Uploading {folder}...")
        
        # We manually iterate through the files and use upload_file to bypass any gitignore logic completely
        for root, _, files in os.walk(folder):
            for file in files:
                local_path = os.path.join(root, file)
                # path in repo should use forward slashes
                repo_path = local_path.replace(os.sep, "/")
                
                print(f"  Uploading {repo_path}...")
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                except Exception as e:
                    print(f"  Failed to upload {repo_path}: {e}")

if __name__ == "__main__":
    main()
