import torch
import gc
import sys
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from src.config import (
    MODEL_PATH, VECTOR_STORE_DIR, COLLECTION_EVIDENCE, COLLECTION_LEGAL
)
from src.ingestion import get_embedding_model

def load_llm():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[INFO] Carregando Llama 3 de: {MODEL_PATH}")
    print(f"       Hardware: CPU (Xeon)")

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=0,
        n_ctx=4096,
        n_batch=64,
        n_threads=8,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.2,
        max_tokens=1024,

        # Streaming DESATIVADO → invoke funciona
        streaming=False,

        # callbacks REMOVIDOS, porque streaming=False
        callbacks=None,
        verbose=False,
    )

    return llm

def get_rag_chain():
    embedding_model = get_embedding_model()

    llm = load_llm()   # <--- FALTAVA ISSO AQUI !!!

    vectorstore_ev = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_EVIDENCE,
        embedding_function=embedding_model
    )

    vectorstore_leg = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=COLLECTION_LEGAL,
        embedding_function=embedding_model
    )

    retriever_ev = vectorstore_ev.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1}
    )

    retriever_leg = vectorstore_leg.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    def combine_contexts(query):
        print("\n[BUSCA] Recuperando evidências e leis no banco vetorial...")
        docs_ev = retriever_ev.invoke(query)
        docs_leg = retriever_leg.invoke(query)

        print(f"[STATUS] Encontrados: {len(docs_ev)} e-mails e {len(docs_leg)} leis.")
        print("[PROCESSAMENTO] Gerando resposta... (O texto deve aparecer abaixo)\n")

        ctx = ""

        for d in docs_ev:
            ctx += f"[EMAIL] Fonte: {d.metadata.get('source')}\n{d.page_content}\n\n"

        for d in docs_leg:
            ctx += f"[LEI] Fonte: {d.metadata.get('source')}\n{d.page_content}\n\n"

        return ctx if ctx else "Nenhum documento encontrado."

    prompt = PromptTemplate.from_template("""
Use APENAS o contexto abaixo.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
""")

    chain = (
        {
            "context": lambda x: combine_contexts(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def run_diagnostics(query):
    print(f"\n[DIAGNOSTICO] Testando recuperação para: '{query}'")
    try:
        embedding_model = get_embedding_model()
        vectorstore_ev = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_EVIDENCE,
            embedding_function=embedding_model
        )
        print("\n1. Teste de Busca (Similarity Search):")
        results = vectorstore_ev.similarity_search(query, k=3)
        if results:
            print(f"   [SUCESSO] Encontrados {len(results)} documentos.")
            for i, doc in enumerate(results):
                src = doc.metadata.get('source', 'N/A')
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   {i+1}. {src}: \"{preview}...\"")
        else:
            print("   [FALHA] Nenhum documento retornado.")
    except Exception as e:
        print(f"   [ERRO CRITICO] {e}")

