import os
os.environ["SKIP_TORCH_CUDA_TEST"] = "1"
os.environ["SKIP_PYTHON_VERSION_CHECK"] = "1"
os.environ["SKIP_INSTALL"] = "1"
os.environ["COMMANDLINE_ARGS"] = "--skip-prepare-environment --enable-insecure-extension-access --xformers"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 离线模式
os.environ['HF_HUB_OFFLINE'] = '1'        # 禁用HuggingFace Hub连接
os.environ['NO_INTERNET'] = '1'           # 告诉A1111不要尝试下载


from webui import webui

if __name__ == "__main__":
    print("Starting WebUI with extensions...")
    webui()
