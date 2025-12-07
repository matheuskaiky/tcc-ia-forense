import os
import requests
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
MODEL_REPO = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
LEGAL_DATASET_ID = "joelniklaus/brazilian_court_decisions"

def download_file(url, dest_path):
    print(f"[INFO] Baixando {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
    print(f"[INFO] Download concluído: {dest_path}")

def setup_enron():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    maildir_path = DATA_RAW_DIR / "maildir"
    tar_path = DATA_RAW_DIR / "enron.tar.gz"

    if maildir_path.exists():
        print("[INFO] Dataset Enron (pasta maildir) já existe. Pulando.")
        return

    if not tar_path.exists():
        download_file(ENRON_URL, tar_path)
    
    print("[INFO] Extraindo Enron Dataset (isso pode demorar)")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=DATA_RAW_DIR)
        print("[INFO] Extração concluída.")
        
        os.remove(tar_path)
    except Exception as e:
        print(f"[ERRO] Falha na extração: {e}")

def setup_legal():
    print(f"[INFO] Verificando dataset de jurisprudencia ({LEGAL_DATASET_ID})...")
    try:
        from datasets import load_dataset
        # streaming=False força o download completo para o cache local do Hugging Face
        load_dataset(LEGAL_DATASET_ID, split="train", streaming=False)
        print("[INFO] Jurisprudencia baixada e cacheada com sucesso.")
    except ImportError:
        print("[ERRO] Biblioteca 'datasets' nao encontrada. Instale com: pip install datasets")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar jurisprudencia: {e}")

def setup_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / MODEL_FILENAME
    
    if destination.exists():
        print(f"[INFO] Modelo {MODEL_FILENAME} já existe. Pulando.")
        return

    print(f"[INFO] Baixando modelo Llama 3 ({MODEL_FILENAME}) via Hugging Face...")
    try:
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print("[INFO] Modelo baixado com sucesso.")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar modelo: {e}")

if __name__ == "__main__":
    print("--- INICIANDO DOWNLOAD DE RECURSOS ---")
    setup_enron()
    setup_legal()
    setup_model()
    print("--- PROCESSO FINALIZADO ---")