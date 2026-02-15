#!/bin/bash
set -Eeuo pipefail

# If existing, delete v2e conda environment
if conda env list | grep -q "^v2e "; then
    echo "Removing v2e conda environment..."
    conda env remove -n v2e -y
else
    echo "v2e conda environment not found, skipping..."
fi

# Remove symbolink links
sudo rm /usr/local/bin/v2e
sudo rm /usr/local/bin/frames2events_quick
