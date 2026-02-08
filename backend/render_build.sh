
#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install CPU-only Torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
python warmup.py
