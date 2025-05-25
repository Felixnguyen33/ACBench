export HF_ENDPOINT=https://hf-mirror.com
 
mkdir -p /data2/share/Qwen2.5

huggingface-cli download --local-dir-use-symlinks False --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir /data2/share/Qwen2.5/Qwen2.5-7B-Instruct
huggingface-cli download --local-dir-use-symlinks False --resume-download mistralai/Mistral-7B-Instruct-v0.3 --local-dir /data2/share/mistral-7B/Mistral-7B-Instruct-v0.3
