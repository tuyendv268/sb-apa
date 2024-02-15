#### GMM Aligner and Nnet3 Aligner
```
sudo nvidia-docker run -it --gpus '"device=0,1"' --cpus 16 \
    -p 9999:9999 \
    -v /data/codes/sb-apa/train:/data/codes/sb-apa/train \
    prep/pykaldi-gpu-python3.9:latest \
    /bin/bash
```
