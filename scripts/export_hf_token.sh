# Exports HF_TOKEN from the Hugging Face CLI token file.
# Usage: source scripts/export_hf_token.sh   (or . scripts/export_hf_token.sh)
if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
