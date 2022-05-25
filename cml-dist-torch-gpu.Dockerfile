FROM docker.repository.cloudera.com/cdsw/ml-runtime-jupyterlab-python3.8-cuda:2021.02.1-b2

# Upgrade packages in the base image
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get -y install telnet wget build-essential cmake && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# RUN wget -nv https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/libnccl-dev_2.7.8-1+cuda11.1_amd64.deb && \
#   wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-cudart-dev-11-1_11.1.74-1_amd64.deb && \
#   wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-driver-dev-11-1_11.1.74-1_amd64.deb && \
#   dpkg -i libnccl-dev_2.7.8-1+cuda11.1_amd64.deb cuda-driver-dev-11-1_11.1.74-1_amd64.deb cuda-cudart-dev-11-1_11.1.74-1_amd64.deb

RUN wget -nv https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz && \
  tar zxf openmpi-3.1.2.tar.gz && \
  cd openmpi-3.1.2 && \
  ./configure --enable-mpirun-prefix-by-default && \
  make -j && make install && \
  cd .. && rm -rf openmpi-3.1.2 openmpi-3.1.2.tar.gz

RUN wget -nv https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run && \
  chmod a+x cuda_11.1.1_455.32.00_linux.run && \
  bash cuda_11.1.1_455.32.00_linux.run --silent  --toolkit --samples --samplespath=/opt/cuda-sample && \
  rm -f cuda_11.1.1_455.32.00_linux.run

RUN echo "cd /usr/lib/x86_64-linux-gnu" >> /usr/local/bin/init-hvd.sh && \
  echo "ln -s libnvidia-ml.so.455.32.00 libnvidia-ml.so.1" >> /usr/local/bin/init-hvd.sh && \
  echo "ln -s libnvidia-ml.so.1 libnvidia-ml.so" >> /usr/local/bin/init-hvd.sh && \
  echo "ln -s libnvidia-opencl.so.455.32.00 libnvidia-opencl.so.1" >> /usr/local/bin/init-hvd.sh  && \
  echo "ln -s libnvidia-ptxjitcompiler.so.455.32.00 libnvidia-ptxjitcompiler.so.1" >> /usr/local/bin/init-hvd.sh && \
  echo "ln -s libnvidia-ptxjitcompiler.so.1 libnvidia-ptxjitcompiler.so" >> /usr/local/bin/init-hvd.sh && \
  echo "touch /usr/local/hvd-initialized" >> /usr/local/bin/init-hvd.sh && \
  echo ldconfig >> /usr/local/bin/init-hvd.sh && chmod a+x /usr/local/bin/init-hvd.sh && \
  echo /usr/local/lib > /etc/ld.so.conf.d/local.conf && ldconfig

COPY horovod-0.19.5-cp38-cp38-linux_x86_64.whl /build
RUN pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html && \
  pip3 install horovod-0.19.5-cp38-cp38-linux_x86_64.whl && \
  pip3 install tensorboardX && \
  pip3 cache purge && horovodrun --check-build

# RUN HOROVOD_CUDA_INCLUDE=/usr/local/cuda/include HOROVOD_CUDA_LIB=/usr/local/cuda/targets/x86_64-linux/lib \
#     HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu \
#     HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_GPU=CUDA HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_CPU_OPERATIONS=MPI \
#     HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 \
#     pip3 install --no-cache-dir horovod==0.19.5 && \
#   pip3 install tensorboardX && \
#   pip3 cache purge && horovodrun --check-build

# Override Runtime label and environment variables metadata
ENV ML_RUNTIME_EDITION="Distributed PyTorch with GPU" \
 ML_RUNTIME_SHORT_VERSION="1.1" \
 ML_RUNTIME_MAINTENANCE_VERSION="1"
ENV ML_RUNTIME_FULL_VERSION="$ML_RUNTIME_SHORT_VERSION.$ML_RUNTIME_MAINTENANCE_VERSION" \
  ML_RUNTIME_DESCRIPTION="This runtime includes the package required to run distrubted model training for PyTorch 1.8.2 LTS, especially Horovod 0.19.5"

LABEL com.cloudera.ml.runtime.edition=$ML_RUNTIME_EDITION \
com.cloudera.ml.runtime.full-version=$ML_RUNTIME_FULL_VERSION \
com.cloudera.ml.runtime.short-version=$ML_RUNTIME_SHORT_VERSION \
com.cloudera.ml.runtime.maintenance-version=$ML_RUNTIME_MAINTENANCE_VERSION \
com.cloudera.ml.runtime.description=$ML_RUNTIME_DESCRIPTION
