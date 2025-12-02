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

# Configurações do Modelo
MODEL_NAME = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
# Tenta pegar do .env, senão usa o padrão
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / MODEL_NAME))

# Configurações de Embeddings e Banco
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_EVIDENCE = "evidence_store"
COLLECTION_LEGAL = "legal_store"

# Garante que diretórios existam
for d in [DATA_DIR, RAW_DATA_DIR, VECTOR_STORE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
