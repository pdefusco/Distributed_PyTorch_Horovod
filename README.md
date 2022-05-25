# How to build the Docker image with CUDA

docker build --network host -t zelkey/cml-dist-openmpi:1.0-cuda11.1 . -f cml-dist-cuda111.Dockerfile
docker run --network host -it zelkey/cml-dist-openmpi:1.0-cuda11.1 /bin/bash

docker build --network host -t zelkey/cml-dist-torch:1.0-cuda11.1 . -f cml-dist-torch-gpu.Dockerfile
docker run --network host -it zelkey/cml-dist-torch:1.0-cuda11.1 /bin/bash
