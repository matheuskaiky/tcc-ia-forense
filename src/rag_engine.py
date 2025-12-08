import torch
import gc
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    MODEL_PATH, MODEL_CONTEXT_SIZE, DEFAULT_MODEL,
    VECTOR_STORE_DIR, COLLECTION_EVIDENCE, COLLECTION_LEGAL
)
from src.ingestion import get_embedding_model

def load_llm():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[INFO] Carregando modelo: {DEFAULT_MODEL}")
    print(f"       Caminho: {MODEL_PATH}")
    print(f"       Hardware: CPU (Xeon)")

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=0,
        n_ctx=MODEL_CONTEXT_SIZE,
        n_batch=512,  # Aumentado para melhor throughput
        n_threads=8,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=2048,

        stop=[
            "<|endoftext|>",   # Fim padrão do Qwen/Llama
            "<|im_end|>",      # Fim de turno de chat
            "PERGUNTA:",       # Evita que ele invente uma nova pergunta (seu caso atual)
            "Human:",          # Evita alucinação de chat
            "\n\n---"          # Para se tentar criar separadores infinitos
        ],

        streaming=False,
        callbacks=None,
        verbose=False,
    )

    return llm

def get_rag_chain():
    embedding_model = get_embedding_model()
    llm = load_llm()

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
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    retriever_leg = vectorstore_leg.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    def combine_contexts(query):
        print("\n[BUSCA] Recuperando evidências e leis no banco vetorial...")
        docs_ev = retriever_ev.invoke(query)
        docs_leg = retriever_leg.invoke(query)

        print(f"[STATUS] Encontrados: {len(docs_ev)} e-mails e {len(docs_leg)} leis.")
        print("[PROCESSAMENTO] Gerando resposta...\n")

        ctx = ""

        if docs_ev:
            ctx += "=== EVIDÊNCIAS (E-MAILS) ===\n\n"
            for i, d in enumerate(docs_ev, 1):
                sender = d.metadata.get('sender', 'Desconhecido')
                subject = d.metadata.get('subject', 'Sem assunto')
                date = d.metadata.get('date', 'Data desconhecida')
                
                ctx += f"E-MAIL {i}:\n"
                ctx += f"Remetente: {sender}\n"
                ctx += f"Assunto: {subject}\n"
                ctx += f"Data: {date}\n"
                ctx += f"Conteúdo:\n{d.page_content}\n"
                ctx += "-" * 50 + "\n\n"
        
        if docs_leg:
            ctx += "=== BASE LEGAL ===\n\n"
            for i, d in enumerate(docs_leg, 1):
                ctx += f"TEXTO LEGAL {i}:\n{d.page_content}\n\n"

        return ctx if ctx else "Nenhum documento relevante encontrado."

    # Prompt otimizado para Qwen/Sabiá (melhor em português)
    prompt = PromptTemplate.from_template("""Você é um assistente especializado em análise forense de e-mails corporativos.

**INSTRUÇÕES CRÍTICAS:**
1. Analise APENAS as informações presentes no CONTEXTO abaixo
2. Cite sempre o número do e-mail ao mencionar informações (Ex: "Conforme E-MAIL 1...")
3. Se não houver evidências suficientes, responda: "Não foram encontradas evidências sobre [tema] nos documentos analisados"
4. NÃO invente nomes, datas ou eventos que não estejam explicitamente no contexto
5. Para análises forenses, mencione: remetentes, destinatários, datas e conteúdo relevante

CONTEXTO:
{context}

PERGUNTA: {question}

ANÁLISE FORENSE:""")

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
    print(f"[MODELO] Usando: {DEFAULT_MODEL}\n")
    
    try:
        embedding_model = get_embedding_model()
        vectorstore_ev = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_EVIDENCE,
            embedding_function=embedding_model
        )
        
        print("=" * 70)
        print("TESTE DE BUSCA (Similarity Search)")
        print("=" * 70)
        
        results = vectorstore_ev.similarity_search(query, k=3)
        
        if results:
            print(f"\n✓ Encontrados {len(results)} documentos relevantes:\n")
            for i, doc in enumerate(results, 1):
                print(f"┌─ DOCUMENTO {i} " + "─" * 55)
                print(f"│ Arquivo: {doc.metadata.get('source', 'N/A')}")
                print(f"│ Remetente: {doc.metadata.get('sender', 'N/A')}")
                print(f"│ Assunto: {doc.metadata.get('subject', 'N/A')}")
                print(f"│ Data: {doc.metadata.get('date', 'N/A')}")
                print(f"│")
                preview = doc.page_content[:200].replace('\n', ' ')
                print(f"│ Conteúdo: {preview}...")
                print(f"└" + "─" * 68 + "\n")
        else:
            print("✗ Nenhum documento retornado.")
            
    except Exception as e:
        print(f"✗ ERRO: {e}")