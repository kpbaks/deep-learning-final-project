#!/usr/bin/env bash

# Check that pixi is installed
if [ ! command -v pixi &> /dev/null ]
then
    if [ command -v cargo &> /dev/null ]
    then
        cargo install --locked pixi
    else
        curl -fsSL https://pixi.sh/install.sh | bash
    fi
fi

if [ -d ~/.pixi/bin ]; then
    export PATH=~/.pixi/bin:$PATH
fi

pixi install

# pixi run python3 -m pip install audio-diffusion-pytorch # Not in conda-forge

echo "Dependencies installed"
