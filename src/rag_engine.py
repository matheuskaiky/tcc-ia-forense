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

    # Define o handler de streaming
    stream_handler = StreamingStdOutCallbackHandler()
    
    n_gpu = -1 if torch.cuda.is_available() else 0
    
    print(f"[INFO] Carregando Llama 3 de: {MODEL_PATH}")
    print(f"       Hardware: {'GPU (CUDA)' if n_gpu == -1 else 'CPU (Xeon)'}")

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=n_gpu,
            n_ctx=8192,
            
            # --- CONFIGURAÇÃO OTIMIZADA PARA CPU ---
            n_batch=512,        
            n_threads=8,        # Ajustado para seus 8 cores
            temperature=0.1,    # Baixa criatividade para evitar alucinação
            top_p=0.90,
            repeat_penalty=1.2, 
            max_tokens=2048,
            # ---------------------------------------
            
            # CORREÇÃO CRÍTICA AQUI:
            callbacks=[stream_handler], # Usa 'callbacks' (lista) em vez de callback_manager
            verbose=False,
            streaming=True
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
        print("[PROCESSAMENTO] Gerando resposta... (O texto deve aparecer abaixo)\n")
        
        ctx = ""
        if docs_ev:
            ctx += "--- EVIDENCIAS DOS EMAILS ---\n"
            for d in docs_ev:
                src = d.metadata.get('source', 'Desconhecido')
                # Limpeza simples de quebras de linha para economizar contexto
                clean_content = d.page_content.replace('\n', ' ')
                ctx += f"Fonte: {src}\nConteúdo: {clean_content}\n\n"
        
        if docs_leg:
            ctx += "--- JURISPRUDENCIA BRASILEIRA ---\n"
            for d in docs_leg:
                src = d.metadata.get('source', 'Desconhecido')
                clean_content = d.page_content.replace('\n', ' ')
                ctx += f"Decisão: {src}\nConteúdo: {clean_content}\n\n"
        
        return ctx if ctx else "Nenhum documento encontrado."

    # Prompt direto e sem tokens especiais manuais para evitar conflito
    template = """Você é um perito forense digital. Use APENAS o contexto abaixo para responder à pergunta.
    
    REGRAS:
    1. Responda em Português.
    2. Cite o nome do arquivo ou decisão (ex: Fonte: 123.) para cada afirmação.
    3. Se não houver informação no contexto, diga "Não encontrei evidências".
    
    CONTEXTO:
    {context}
    
    PERGUNTA: {question}
    
    RELATÓRIO FORENSE:
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