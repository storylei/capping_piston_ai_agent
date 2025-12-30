#!/bin/bash

# ===== Basic Environment Setup =====
# Set default Python3 to /usr/local/bin/python
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python 1
sudo update-alternatives --set python3 /usr/local/bin/python

# ===== Install Git LFS =====
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install  # Initialize LFS

# ===== Python Toolchain =====
# Install core packages without dependencies to avoid conflicts
python3 -m pip install --upgrade --no-deps notebook ipykernel

# Register Jupyter kernel with descriptive name
python3 -m ipykernel install --name=python3 --display-name="Python 3.12 (System)"

# ===== Project Dependencies =====
# Install packages from requirements.txt
python3 -m pip install -r requirements.txt

# ===== Permission Fixes =====
# Ensure proper ownership of user directories
sudo chown -R ${USER}:${USER} /home/${USER}/.local

# ===== Start Ollama Service =====
# Start Ollama in background (for non-systemd environments like Codespaces)
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 2  # Give Ollama time to start
echo "✅ Ollama service started (background process)"

# ===== Verification Output =====
echo "=== Environment Setup Complete ==="
echo "Git LFS Status:"
git lfs env | grep "Git LFS initialized"
echo "Python Environment:"
python3 -c "import sys; print(f'Python Path: {sys.executable}')"
echo "Ollama Status:"
ollama --version || echo "⚠️  Ollama not yet ready, will start on demand"