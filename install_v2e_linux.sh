python3.11 -m venv .venvEventBased
source .venvEventBased/bin/activate
python3.11 -m pip install -e . --require-virtualenv
pip install torch torchvision --require-virtualenv
pip install dv-processing pillow==9.5.0 --require-virtualenv

# Add symlink to v2e.py in system location
sudo ln -s $HOME/devDir/event-based-repos/v2e_PeterCdev/v2e.py /usr/local/bin/v2e
sudo chmod +xr v2e

# Add symlink to frames2events_quick.sh in system location
sudo ln -s $HOME/devDir/event-based-repos/v2e_PeterCdev/frames2events_quick.sh /usr/local/bin/frames2events_quick
sudo chmod +xr frames2events_quick