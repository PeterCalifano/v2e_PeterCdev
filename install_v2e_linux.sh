python3.11 -m venv .venvEventBased
source .venvEventBased/bin/activate
python3.11 -m pip install -e . --require-virtualenv
pip install torch torchvision --require-virtualenv
pip install dv-processing pillow==9.5.0 --require-virtualenv

# Install tkinter for python3.11 and required deps
sudo apt-get update
sudo apt-get install build-essential libbz2-dev libssl-dev libreadline-dev libsqlite3-dev zlib1g-dev libffi-dev libncurses5-dev libgdbm-dev liblzma-dev tk-dev libtk8.6
sudo apt install python3.11-tk

# Add symlink to v2e.py in system location
sudo ln -s $HOME/devDir/event-based-repos/v2e_PeterCdev/v2e.py /usr/local/bin/v2e
sudo chmod +xr /usr/local/bin/v2e

# Add symlink to frames2events_quick.sh in system location
sudo ln -s $HOME/devDir/event-based-repos/v2e_PeterCdev/frames2events_quick.sh /usr/local/bin/frames2events_quick
sudo chmod +xr /usr/local/bin/frames2events_quick