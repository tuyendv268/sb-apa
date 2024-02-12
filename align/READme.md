#### GMM Aligner and Nnet3 Aligner
```
sudo nvidia-docker run -it --gpus '"device=0,1"' --cpus 0.6 \
    -p 9999:9999 \
    -v /data/codes/apa/:/workspace \
    -v /data/audio/prep-submission-audio/:/data/audio/prep-submission-audio/ \
    prep/pykaldi-gpu-python3.9:latest \
    /bin/bash
```
