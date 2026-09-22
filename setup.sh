apt-get update && apt-get install -y git build-essential cmake
apt-get update && apt-get install -y nodejs npm
npm install -g pm2

cd /workspace

CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86" pip install llama-cpp-python[server]

pip install fastapi supabase pylatexenc pandas sentence_transformers cohere openai huggingface-hub
pip install -U google-genai==1.60.0

cd /workspace
pm2 start server.py --name deepseek-server -- --config_file /workspace/deepseek.json

pm2 start server.py --name llama-server -- --config_file /workspace/llama.json

pm2 start server.py --name qwen-server -- --config_file /workspace/qwen.json
