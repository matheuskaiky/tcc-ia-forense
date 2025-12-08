import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Caminhos do Projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
MODELS_DIR = BASE_DIR / "models"

# ============================================
# CONFIGURAÇÃO DE MODELOS DISPONÍVEIS
# ============================================

AVAILABLE_MODELS = {
    "llama3": {
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "n_ctx": 4096,
        "prompt_format": "llama3"
    },
    "qwen2.5-7b": {
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "n_ctx": 8192,
        "prompt_format": "chatml"
    },
    "qwen2.5-14b": {
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "n_ctx": 8192,
        "prompt_format": "chatml"
    },
    "sabia3": {
        "filename": "sabia-3-8b-instruct-Q4_K_M.gguf",
        "n_ctx": 8192,
        "prompt_format": "chatml"
    }
}

# Modelo padrão (pode ser alterado via .env: MODEL_CHOICE=qwen2.5-14b)
DEFAULT_MODEL = os.getenv("MODEL_CHOICE", "qwen2.5-14b")

# Valida se o modelo existe
if DEFAULT_MODEL not in AVAILABLE_MODELS:
    print(f"[AVISO] Modelo '{DEFAULT_MODEL}' não encontrado. Usando 'llama3'.")
    DEFAULT_MODEL = "llama3"

MODEL_CONFIG = AVAILABLE_MODELS[DEFAULT_MODEL]
MODEL_NAME = MODEL_CONFIG["filename"]
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / MODEL_NAME))
MODEL_CONTEXT_SIZE = MODEL_CONFIG["n_ctx"]
MODEL_PROMPT_FORMAT = MODEL_CONFIG["prompt_format"]

# Configurações de Embeddings e Banco
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_EVIDENCE = "evidence_store"
COLLECTION_LEGAL = "legal_store"

# Garante que diretórios existam
for d in [DATA_DIR, RAW_DATA_DIR, VECTOR_STORE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)