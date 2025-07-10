#!/bin/bash

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

python data.py