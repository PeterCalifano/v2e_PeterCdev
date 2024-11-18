#python3.11 -m venv .venvEventBased
#source .venvEventBased/bin/activate
python3.11 -m pip install -e . --require-virtualenv
pip install torch torchvision --require-virtualenv
pip install dv-processing --require-virtualenv