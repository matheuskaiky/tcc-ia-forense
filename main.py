import argparse
import sys
from src.ingestion import ingest_evidence, ingest_legal

def interactive_mode():
    """
    Inicia um loop de chat contínuo com o sistema.
    """
    from src.rag_engine import get_rag_chain
    
    print("\n[INFO] Carregando sistema RAG e modelo Llama 3...")
    try:
        chain = get_rag_chain()
        print("[INFO] Sistema pronto. Digite 'sair' para encerrar.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nPergunta: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    print("[INFO] Encerrando sistema.")
                    break
                
                print("-" * 50)
                response = chain.invoke({"question": user_input})
                print(response)
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\n[INFO] Interrupção detectada. Encerrando.")
                break
            except Exception as e:
                print(f"[ERRO] Falha ao processar pergunta: {e}")

    except Exception as e:
        print(f"[ERRO] Falha crítica ao carregar o sistema: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="TCC Forense AI - Sistema RAG Local")
    
    parser.add_argument("--ingest", action="store_true", help="Executa a ingestão de dados (E-mails e Leis)")
    parser.add_argument("--query", type=str, help="Faz uma pergunta única ao sistema e encerra")
    parser.add_argument("--interactive", action="store_true", help="Inicia o modo de chat interativo")
    parser.add_argument("--diagnose", type=str, help="Executa testes de recuperação para uma query específica")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    if args.ingest:
        print("[INFO] Iniciando processo de ingestão de dados...")
        try:
            print("[1/2] Processando Jurisprudência...")
            ingest_legal(limit_docs=200)
            print("[2/2] Processando Evidências (Dataset Enron)...")
            ingest_evidence(limit_files=500)
            print("[INFO] Ingestão concluída.")
        except Exception as e:
            print(f"[ERRO] Falha na ingestão: {e}")

    if args.diagnose:
        from src.rag_engine import run_diagnostics
        print(f"[INFO] Iniciando diagnóstico para a query: '{args.diagnose}'")
        run_diagnostics(args.diagnose)
        return

    if args.interactive:
        interactive_mode()
        return
    
    if args.query:
        print(f"\n[INFO] Pergunta Recebida: {args.query}")
        print("-" * 50)
        
        try:
            from src.rag_engine import get_rag_chain
            chain = get_rag_chain()
            
            resposta = chain.invoke({"question": args.query})
            
            if not resposta: 
                print("[AVISO] A resposta retornou vazia.")
            else:
                print("\n") 
            
            print("-" * 50)
        except Exception as e:
            print(f"[ERRO] Falha na execução: {e}")

if __name__ == "__main__":
    main()