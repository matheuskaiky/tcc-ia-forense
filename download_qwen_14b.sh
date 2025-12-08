#!/bin/bash

# Script para baixar e juntar Qwen 2.5 14B corretamente

set -e  # Para no primeiro erro

echo "================================================"
echo "Download e montagem do Qwen 2.5 14B"
echo "================================================"

cd models/

# Limpar downloads anteriores
echo "[1/5] Limpando arquivos anteriores..."
rm -f qwen2.5-14b-instruct-q4_k_m*.gguf

# Baixar as 3 partes
echo "[2/5] Baixando parte 1/3 (2.9GB)..."
wget -c https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf

echo "[3/5] Baixando parte 2/3 (2.9GB)..."
wget -c https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf

echo "[4/5] Baixando parte 3/3 (2.7GB)..."
wget -c https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf

# Juntar os arquivos
echo "[5/5] Juntando arquivos..."
cat qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
    qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf \
    qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf \
    > qwen2.5-14b-instruct-q4_k_m.gguf

# Verificar tamanho
FILESIZE=$(stat -f%z qwen2.5-14b-instruct-q4_k_m.gguf 2>/dev/null || stat -c%s qwen2.5-14b-instruct-q4_k_m.gguf)
EXPECTED=8500000000  # ~8.5GB

echo ""
echo "================================================"
echo "Verificação de Integridade"
echo "================================================"
echo "Tamanho do arquivo: $(numfmt --to=iec-i --suffix=B $FILESIZE 2>/dev/null || echo $FILESIZE bytes)"
echo "Tamanho esperado:   ~8.5GB"

if [ $FILESIZE -gt 8000000000 ]; then
    echo "✓ Arquivo parece estar completo!"
    
    # Remover partes
    echo ""
    echo "Removendo arquivos temporários..."
    rm -f qwen2.5-14b-instruct-q4_k_m-0000*-of-00003.gguf
    
    echo ""
    echo "================================================"
    echo "✓ DOWNLOAD CONCLUÍDO COM SUCESSO!"
    echo "================================================"
    echo "Arquivo final: qwen2.5-14b-instruct-q4_k_m.gguf"
    echo ""
    echo "Configure no .env:"
    echo "  echo 'MODEL_CHOICE=qwen2.5-14b' > ../.env"
else
    echo "✗ AVISO: Arquivo parece estar incompleto!"
    echo "  Execute o script novamente."
    exit 1
fi
