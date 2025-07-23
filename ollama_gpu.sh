# Stop Ollama
sudo systemctl stop ollama

# Start with GPU layers
OLLAMA_NUM_GPU=99 ollama serve