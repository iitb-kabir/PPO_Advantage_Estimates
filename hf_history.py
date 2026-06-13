from huggingface_hub import HfApi
api = HfApi()
commits = api.list_repo_commits(repo_id="kabiriitb/ppo_advantage_estimator")
for i, c in enumerate(commits):
    print(f"{i}: {c.commit_id} - {c.title}")
