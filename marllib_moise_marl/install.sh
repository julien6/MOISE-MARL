#!/usr/bin/env bash

git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
git config --global core.compression 0

# -----------------------------------
# Vérification Python 3.8
# -----------------------------------
if command -v python3.8 >/dev/null 2>&1; then
    echo "✅ Python 3.8 was found."
else
    echo "❌ Python 3.8 is not installed. Please install it (ex: sudo apt install python3.8 python3.8-venv)."
    exit 1
fi

echo "🛠️  Creating virtual Python 3.8 environment..."
python3.8 -m venv ~/mma

echo "Activation de l'environnement virtuel..."
source ~/mma/bin/activate

pip install --upgrade pip setuptools

git clone https://github.com/julien6/PettingZoo.git

cd PettingZoo

pip install -e .

cd ..

git clone https://github.com/julien6/MARLlib.git
cd MARLlib
pip install --upgrade pip
pip install setuptools==65.5.0 pip==21 ; pip install wheel==0.38.0 # gym 0.21 installation is broken with more recent versions
pip install -r requirements.txt

# we recommend the gym version between 0.20.0~0.22.0.
pip install "gym>=0.20.0,<0.22.0"

pip install numpy==1.20.3
# pip install pettingzoo==1.12.0
pip install pyglet==1.5.11

cd marllib
# add patch files to MARLlib
python patch/add_patch.py -y

cd ..

pip install -e .

cd ../.

# Install Warehouse Management and Moving Company
git clone https://github.com/julien6/OMARLE.git
cd OMARLE
pip install -e .

cd ..

# Install OvercookedAI
git clone https://github.com/julien6/overcooked_ai.git
cd overcooked_ai
pip install -e .

cd ..

# Install CybORG
git clone https://github.com/julien6/CybORG.git
cd CybORG
pip install -e .

cd ..

cd mma
pip install -e .

cd ../..
pip install -r requirements.txt
