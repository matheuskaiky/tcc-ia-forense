import os
import email
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
import torch
from src.config import (
    RAW_DATA_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME,
    COLLECTION_EVIDENCE, COLLECTION_LEGAL
)

def get_embedding_model():
    # Detecta GPU (Cuda ou MPS para Mac)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Carregando Embeddings em: {device.upper()}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def parse_enron_email(file_path):
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            msg = email.message_from_file(f)
            content = msg.get_payload()
            metadata = {
                "source": os.path.basename(file_path),
                "sender": msg.get("From", "Unknown"),
                "subject": msg.get("Subject", "No Subject"),
                "date": msg.get("Date", "Unknown"),
                "type": "email_evidence"
            }
            return Document(page_content=content, metadata=metadata)
    except Exception:
        return None

def ingest_evidence(limit_files=500):
    embedding_model = get_embedding_model()
    documents = []
    file_count = 0
    maildir = RAW_DATA_DIR / 'maildir'

    if not maildir.exists():
        print(f"ERRO: Pasta {maildir} não encontrada.")
        print("   -> Baixe o dataset Enron e coloque a pasta 'maildir' dentro de 'data/raw/'")
        return

    print(f"Lendo arquivos de evidência em: {maildir}")
    for root, _, files in os.walk(maildir):
        for file in files:
            if file_count >= limit_files: break
            
            file_path = os.path.join(root, file)
            if not file.startswith('.'):
                doc = parse_enron_email(file_path)
                if doc and len(doc.page_content) > 10:
                    documents.append(doc)
                    file_count += 1
        if file_count >= limit_files: break

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    
    if splits:
        print(f"Salvando {len(splits)} chunks no ChromaDB ({COLLECTION_EVIDENCE})...")
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_EVIDENCE
        )
        print("Ingestão de Evidências concluída.")
    else:
        print("Nenhum documento processado.")

def ingest_legal(limit_docs=200):
    embedding_model = get_embedding_model()
    print("Baixando jurisprudência (Streaming do Hugging Face)...")
    
    try:
        ds = load_dataset("joelniklaus/brazilian_court_decisions", split="train", streaming=True)
        legal_docs = []
        
        for i, item in enumerate(ds):
            if i >= limit_docs: break
            content = item.get('decision_description', '')
            if content and len(content) > 100:
                meta = {
                    "source": f"juris_br_id_{item.get('id', i)}",
                    "court": "Tribunais BR",
                    "type": "legal_knowledge"
                }
                legal_docs.append(Document(page_content=content, metadata=meta))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(legal_docs)
        
        print(f"Salvando {len(splits)} chunks no ChromaDB ({COLLECTION_LEGAL})...")
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_LEGAL
        )
        print("Ingestão Legal concluída.")
    except Exception as e:
        print(f"Erro ao baixar jurisprudência: {e}")
