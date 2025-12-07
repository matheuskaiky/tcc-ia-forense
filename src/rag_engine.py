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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    n_gpu = -1 if torch.cuda.is_available() else 0
    
    print(f"[INFO] Carregando Llama 3 de: {MODEL_PATH}")
    print(f"       Layers na GPU: {'TODAS' if n_gpu == -1 else 'CPU (Zero)'}")

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=n_gpu,
            n_ctx=8192,
            temperature=0.6,    # ajuste para evitar recusas excessivas
            top_p=0.95,         # núcleus sampling
            repeat_penalty=1.1, # evita loops
            max_tokens=4096,
            callback_manager=callback_manager,
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"[ERRO] Falha ao carregar LlamaCpp: {e}")
        print("       Verifique se o arquivo .gguf existe em 'models/'")
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
        print("\n[BUSCA] Recuperando evidências e leis no banco vetorial...")
        docs_ev = retriever_ev.invoke(query)
        docs_leg = retriever_leg.invoke(query)
        
        ctx = ""
        if docs_ev:
            ctx += "\n--- EVIDENCIAS (ENRON DATASET) ---\n"
            for d in docs_ev:
                src = d.metadata.get('source', 'Desconhecido')
                clean_content = d.page_content.replace('\n', ' ')
                ctx += f"[Fonte: {src}] {clean_content}\n"
        
        if docs_leg:
            ctx += "\n--- JURISPRUDENCIA ---\n"
            for d in docs_leg:
                src = d.metadata.get('source', 'Desconhecido')
                clean_content = d.page_content.replace('\n', ' ')
                ctx += f"[Decisao: {src}] {clean_content}\n"
        
        if not ctx:
            print("[AVISO] Nenhum documento relevante encontrado.")
            return "Nenhuma informação relevante encontrada nos documentos."
            
        return ctx

    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Você é um Assistente Especialista em Forense Digital e Direito.
    Responda à pergunta do investigador baseando-se APENAS no contexto abaixo.
    
    REGRAS:
    1. Cite estritamente a fonte para cada fato (ex: [Fonte: email_x]).
    2. Se houver infração, cite a jurisprudência correlata.
    3. Responda em Português do Brasil.
    4. Seja técnico e direto.
    
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

def run_diagnostics(query):
    """
    Função de diagnóstico para verificar se o Retrieval está funcionando
    sem precisar carregar o LLM pesado.
    """
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
            print("   [FALHA] Nenhum documento retornado. Verifique a ingestão.")

    except Exception as e:
        print(f"   [ERRO CRITICO] Falha no teste de diagnóstico: {e}")