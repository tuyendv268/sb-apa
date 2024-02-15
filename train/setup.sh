#!/usr/bin/env bash

pip install -r requirements.txt
python -m pip install git+https://github.com/NVIDIA/NeMo-text-processing.git@main#egg=nemo_text_processing

pip install torch==2.1.2 \
            torchvision==0.16.2 \
            torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install "ray[serve]==2.9.0"