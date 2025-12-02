import argparse
import sys
from src.ingestion import ingest_evidence, ingest_legal

def main():
    parser = argparse.ArgumentParser(description="TCC Forense AI - RAG System Local")
    parser.add_argument("--ingest", action="store_true", help="Executa a ingestão de dados e cria o Vector DB")
    parser.add_argument("--query", type=str, help="Faz uma pergunta ao sistema")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    if args.ingest:
        print("🚀 Iniciando processo de ingestão...")
        # Baixa leis
        ingest_legal(limit_docs=200)
        # Processa e-mails (requer que os arquivos existam)
        ingest_evidence(limit_files=500)
    
    if args.query:
        print(f"\n🔎 Pergunta: {args.query}")
        print("-" * 50)
        
        # Importação tardia para não carregar LLM se for só ingestão
        from src.rag_engine import get_rag_chain
        
        chain = get_rag_chain()
        chain.invoke({"question": args.query})
        
        print("\n" + "-" * 50)

if __name__ == "__main__":
    main()
