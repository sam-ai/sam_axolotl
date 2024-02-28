import fire
from huggingface_hub import HfApi

class HuggingFaceUploader:
    def __init__(self):
        self.api = HfApi()

    def upload_folder(self, folder_path, repo_id, repo_type):
        """
        Uploads a folder to the Hugging Face Model Hub.

        Args:
            folder_path (str): Path to the local folder to be uploaded.
            repo_id (str): ID of the repository in the Hugging Face Model Hub (e.g., "username/my-cool-space").
            repo_type (str): Type of the repository (e.g., "space").
        """
        self.api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"Folder '{folder_path}' uploaded to '{repo_id}' ({repo_type}).")

if __name__ == "__main__":
    fire.Fire(HuggingFaceUploader)
