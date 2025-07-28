#!/bin/bash

sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python 1
sudo update-alternatives --set python3 /usr/local/bin/python

python3 -m pip install --upgrade --no-deps notebook ipykernel
python3 -m ipykernel install --name=python3 --display-name="Python 3.12 (System)"

python3 -m pip install -r requirements.txt

sudo chown -R ${USER}:${USER} /home/${USER}/.local