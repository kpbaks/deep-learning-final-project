#!/usr/bin/env bash

# Check that pixi is installed
if [ ! command -v pixi &> /dev/null ]
then
    echo "pixi could not be found"
    echo "Please install pixi and try again"
    exit 1
fi


pixi install

# pixi run python3 -m pip install audio-diffusion-pytorch # Not in conda-forge

echo "Dependencies installed"
