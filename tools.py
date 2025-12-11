from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import huggingface_hub
# huggingface_hub.login("你的Token") 
def download_bge_m3_model(target_dir: str = "./bge-m3"):
    """
    下载 BAAI/bge-m3 模型到指定目录。

    :param target_dir: 下载的目标路径，默认为当前目录下的 bge-m3 文件夹
    """
    snapshot_download(
        repo_id="BAAI/bge-m3",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

# Example usage:
download_bge_m3_model(r"C:\Users\31029\Desktop\Langchain-Chatchat\bge-m3")
