import os
import sys
from huggingface_hub import HfApi

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_hf.py <repo_id>")
        print("Example: python upload_to_hf.py kabiriitb/ppo_advantage_estimator")
        sys.exit(1)
        
    repo_id = sys.argv[1]
    api = HfApi()
    
    print(f"Uploading to {repo_id}...")
    
    # We rename .gitignore temporarily so the Hugging Face API doesn't ignore our results and models
    gitignore_exists = os.path.exists(".gitignore")
    if gitignore_exists:
        os.rename(".gitignore", ".gitignore.bak")
        print("Temporarily disabled .gitignore")
    
    try:
        # Upload everything in the current directory except the git folder
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[".git/**", ".gitignore.bak"]
        )
        print("\nSuccessfully uploaded everything including models, figures, and results to Hugging Face!")
    except Exception as e:
        print(f"\nError during upload: {e}")
    finally:
        # Always restore .gitignore
        if gitignore_exists:
            os.rename(".gitignore.bak", ".gitignore")
            print("Restored .gitignore")

if __name__ == "__main__":
    main()
