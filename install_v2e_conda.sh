#python3.11 -m venv .venvEventBased
#source .venvEventBased/bin/activate
#python3.11 -m pip install -e . --require-virtualenv
#pip install torch torchvision --require-virtualenv
#pip install dv-processing pillow==9.5.0 --require-virtualenv
set -Eeuo pipefail

# Create conda environment
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "v2e"; then
  conda create -y -n v2e python=3.11
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
else
  REPO_ROOT="$SCRIPT_DIR"
fi

# Install tkinter for python3.11 and required deps
sudo apt-get update
sudo apt-get install build-essential libbz2-dev libssl-dev libreadline-dev libsqlite3-dev zlib1g-dev libffi-dev libncurses5-dev libgdbm-dev liblzma-dev tk-dev libtk8.6
sudo apt install python3.11-tk

# Add symlink to v2e.py in system location if not found
if [ ! -L /usr/local/bin/v2e ]; then
  sudo ln -s $REPO_ROOT/v2e.py /usr/local/bin/v2e
  sudo chmod +xr /usr/local/bin/v2e
fi

# Add symlink to frames2events_quick.sh in system location if not found
if [ ! -L /usr/local/bin/frames2events_quick ]; then
  sudo ln -s $REPO_ROOT/frames2events_quick.sh /usr/local/bin/frames2events_quick
  sudo chmod +xr /usr/local/bin/frames2events_quick
fi
