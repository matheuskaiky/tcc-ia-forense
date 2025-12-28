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
    # Limpeza de memória
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Callback para efeito de digitação (Streaming)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    n_gpu = -1 if torch.cuda.is_available() else 0
    
    print(f"[INFO] Carregando Llama 3 de: {MODEL_PATH}")
    print(f"       Modo: {'GPU (CUDA)' if n_gpu == -1 else 'CPU (Xeon)'}")

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=n_gpu,
            n_ctx=8192,
            
            # --- Configuração Otimizada ---
            n_batch=512,
            n_threads=8,
            temperature=0.1,    # Baixa temperatura para precisão
            top_p=0.9,
            repeat_penalty=1.1, 
            max_tokens=2048,
            
            # Stop tokens para evitar que ele alucine novas perguntas
            stop=["<|eot_id|>", "<|end_of_text|>", "PERGUNTA:", "Human:", "AI:"],
            
            callback_manager=callback_manager,
            verbose=False,
            streaming=False
        )
        return llm
    except Exception as e:
        print(f"[ERRO] Falha ao carregar LlamaCpp: {e}")
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
        
        print(f"[STATUS] Encontrados: {len(docs_ev)} e-mails e {len(docs_leg)} leis.")
        print("[PROCESSAMENTO] Gerando Relatório Técnico...\n")
        print("-" * 50)
        
        ctx = ""
        # 1. Bloco de Evidências
        if docs_ev:
            ctx += "=== EVIDÊNCIAS DIGITAIS (DATASET ENRON) ===\n"
            for d in docs_ev:
                src = d.metadata.get('source', 'Desconhecido')
                sender = d.metadata.get('sender', 'N/A')
                # Limpa quebras de linha para economizar tokens
                clean_content = d.page_content.replace('\n', ' ')
                ctx += f"[ARQUIVO: {src} | REMETENTE: {sender}]\nCONTEÚDO: {clean_content}\n\n"
        
        # 2. Bloco Legal (Melhorado para citação)
        if docs_leg:
            ctx += "=== BASE JURÍDICA (JURISPRUDÊNCIA BRASILEIRA) ===\n"
            for d in docs_leg:
                src = d.metadata.get('source', 'N/A')
                court = d.metadata.get('court', 'Tribunal N/A')
                clean_content = d.page_content.replace('\n', ' ')
                # Formatação explícita para o LLM
                ctx += f"[DECISÃO ID: {src} | TRIBUNAL: {court}]\nEMENTA: {clean_content}\n\n"
        
        return ctx if ctx else "Nenhuma informação encontrada."

    # --- NOVO PROMPT LIMPO ---
    template = """<|start_header_id|>system<|end_header_id|>

    Você é um Perito Forense Digital. Sua tarefa é gerar um Relatório Técnico com base EXCLUSIVA nos documentos abaixo.
    
    ESTRUTURA DO RELATÓRIO:
    1. ANÁLISE FORENSE: Cite fatos, nomes, datas e arquivos (ex: "Conforme e-mail X...").
    2. ANÁLISE JURÍDICA: Cite as decisões judiciais recuperadas (ID e Tribunal) e explique se elas se aplicam ou não ao caso.
    
    IMPORTANTE:
    - Se a jurisprudência recuperada não for relevante, diga: "A Decisão [ID] sobre [assunto] não se aplica diretamente pois..."
    - Não invente informações.
    
    DOCUMENTOS RECUPERADOS:
    {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    PERGUNTA DO INVESTIGADOR: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    RELATÓRIO TÉCNICO:
    
    ### 1. ANÁLISE FORENSE (Fatos)
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

# Mantive a função de diagnóstico igual para seus testes
def run_diagnostics(query):
    print(f"\n[DIAGNOSTICO] Testando recuperação para: '{query}'")
    try:
        embedding_model = get_embedding_model()
        vectorstore_ev = Chroma(persist_directory=str(VECTOR_STORE_DIR), collection_name=COLLECTION_EVIDENCE, embedding_function=embedding_model)
        vectorstore_leg = Chroma(persist_directory=str(VECTOR_STORE_DIR), collection_name=COLLECTION_LEGAL, embedding_function=embedding_model)
        
        print("\n--- Evidências ---")
        evs = vectorstore_ev.similarity_search(query, k=2)
        for i, d in enumerate(evs): print(f"{i+1}. {d.metadata.get('source')}")
            
        print("\n--- Leis ---")
        legs = vectorstore_leg.similarity_search(query, k=2)
        for i, d in enumerate(legs): 
            src = d.metadata.get('source')
            court = d.metadata.get('court', 'N/A')
            print(f"{i+1}. {src} ({court})")
            
    except Exception as e:
        print(f"[ERRO] {e}")