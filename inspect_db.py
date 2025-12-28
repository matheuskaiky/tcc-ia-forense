import argparse
import sys
from langchain_chroma import Chroma
from src.config import VECTOR_STORE_DIR, COLLECTION_LEGAL, COLLECTION_EVIDENCE, EMBEDDING_MODEL_NAME
from src.ingestion import get_embedding_model

def inspect(collection_name, doc_id=None, list_all=False):
    print(f"\n🔍 Inspecionando Coleção: {collection_name}")
    print(f"📂 Diretório: {VECTOR_STORE_DIR}")
    
    # Conecta ao banco sem precisar carregar o modelo pesado (usamos embedding fake se precisar, mas aqui o get é direto)
    # Nota: Para leitura de metadados raw, o Chroma native client é mais rápido, 
    # mas vamos usar o LangChain para manter compatibilidade com sua config.
    try:
        embedding_model = get_embedding_model()
        vectorstore = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=collection_name,
            embedding_function=embedding_model
        )
        
        # Acessa a coleção crua do Chroma
        collection = vectorstore._collection
        count = collection.count()
        print(f"📊 Total de documentos na coleção: {count}")
        
        if list_all:
            print("\n📜 Listando todos os documentos (ID e Fonte):")
            # Pega apenas metadados para ser rápido
            data = collection.get(include=['metadatas'])
            for i, meta in enumerate(data['metadatas']):
                source = meta.get('source', 'N/A')
                print(f"   [{i+1}] {source}")
            return

        if doc_id:
            print(f"\n🔎 Buscando documento com source = '{doc_id}'...")
            # O "where" filtra pelos metadados
            data = collection.get(where={"source": doc_id})
            
            if data['ids']:
                print(f"✅ Documento encontrado!")
                print("-" * 50)
                # Pega o primeiro match (IDs são únicos no Chroma, mas source pode não ser)
                print(f"METADADOS: {data['metadatas'][0]}")
                print("-" * 50)
                print(f"CONTEÚDO:\n{data['documents'][0]}")
                print("-" * 50)
            else:
                print("❌ Nenhum documento encontrado com esse ID.")
        else:
            # Se não pediu nada específico, mostra os 5 primeiros como amostra
            print("\n🎲 Amostra dos 5 primeiros documentos:")
            data = collection.get(limit=5)
            for i, (meta, content) in enumerate(zip(data['metadatas'], data['documents'])):
                print(f"\n--- Documento {i+1} ---")
                print(f"ID/Fonte: {meta.get('source', 'N/A')}")
                print(f"Conteúdo (trecho): {content[:200]}...")

    except Exception as e:
        print(f"❌ Erro ao ler banco: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspetor do ChromaDB")
    parser.add_argument("--legal", action="store_true", help="Inspeciona o banco de Leis")
    parser.add_argument("--evidence", action="store_true", help="Inspeciona o banco de Evidências")
    parser.add_argument("--id", type=str, help="Busca um documento específico pelo nome da fonte (ex: juris_br_id_10)")
    parser.add_argument("--list", action="store_true", help="Lista todos os IDs disponíveis")

    args = parser.parse_args()

    if not (args.legal or args.evidence):
        print("⚠️  Use --legal ou --evidence para escolher qual banco inspecionar.")
        sys.exit(1)

    col_name = COLLECTION_LEGAL if args.legal else COLLECTION_EVIDENCE
    inspect(col_name, doc_id=args.id, list_all=args.list)