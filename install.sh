#!/usr/bin/env sh

# Bash script: install_submodules.sh
# Navigate to each submodule directory and install it
for lib in libraries/*; do
    if [ -d "$lib" ]; then
        echo "Installing submodule: $lib"
        pip install -e "$lib"
    fi
done

# Now install the main package
pip install -e .
