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

# ===== Verification Output =====
echo "=== Environment Setup Complete ==="
echo "Git LFS Status:"
git lfs env | grep "Git LFS initialized"
echo "Python Environment:"
python3 -c "import sys; print(f'Python Path: {sys.executable}')"