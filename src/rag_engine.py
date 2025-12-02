import torch
import gc
import sys
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from src.config import (
    MODEL_PATH, VECTOR_STORE_DIR, COLLECTION_EVIDENCE, COLLECTION_LEGAL
)
from src.ingestion import get_embedding_model

def load_llm():
    # Tenta limpar memória
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Configuração de GPU
    n_gpu = -1 if torch.cuda.is_available() else 0
    
    print(f"🤖 Carregando Llama 3 de: {MODEL_PATH}")
    print(f"   -> Layers na GPU: {'TODAS' if n_gpu == -1 else 'CPU (Zero)'}")

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=n_gpu,
            n_ctx=8192,
            temperature=0.0,
            # max_tokens=1024, # ajuste por falta de RAM
            max_tokens=512,
            callback_manager=callback_manager,
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"❌ Erro ao carregar LlamaCpp: {e}")
        print("   -> Verifique se o arquivo .gguf existe em 'models/'")
        sys.exit(1)

def get_rag_chain():
    embedding_model = get_embedding_model()
    
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

    retriever_ev = vectorstore_ev.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    retriever_leg = vectorstore_leg.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    def combine_contexts(query):
        print("\n🔍 Buscando evidências e leis...")
        docs_ev = retriever_ev.invoke(query)
        docs_leg = retriever_leg.invoke(query)
        
        ctx = ""
        if docs_ev:
            ctx += "\n--- EVIDÊNCIAS (EMAILS) ---\n"
            for d in docs_ev:
                src = d.metadata.get('source', 'Desconhecido')
                ctx += f"[Fonte: {src}] " + d.page_content.replace('\n', ' ') + "\n"
        
        if docs_leg:
            ctx += "\n--- JURISPRUDÊNCIA ---\n"
            for d in docs_leg:
                src = d.metadata.get('source', 'Desconhecido')
                ctx += f"[Decisão: {src}] " + d.page_content.replace('\n', ' ') + "\n"
        
        return ctx if ctx else "Nenhuma informação relevante encontrada."

    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Você é um Assistente Especialista em Forense Digital e Direito.
    Responda à pergunta do investigador baseando-se APENAS no contexto abaixo.
    
    REGRAS:
    1. Cite estritamente a fonte para cada fato (ex: [Fonte: email_x]).
    2. Se houver infração, cite a jurisprudência correlata.
    3. Responda em Português.
    
    CONTEXTO:
    {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = PromptTemplate.from_template(template)
    llm = load_llm()

    chain = (
        {"context": lambda x: combine_contexts(x["question"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
