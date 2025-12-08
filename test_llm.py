from llama_cpp import Llama
import os

model_path = "./models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"

if not os.path.exists(model_path):
    print("ERRO: Arquivo não encontrado pelo Python!")
else:
    print(f"Arquivo encontrado. Tamanho: {os.path.getsize(model_path) / (1024*1024*1024):.2f} GB")
    print("Tentando carregar...")
    
    try:
        llm = Llama(
            model_path=model_path,
            verbose=True
        )
        print("SUCESSO! O modelo carregou.")
    except Exception as e:
        print(f"ERRO AO CARREGAR: {e}")