FROM docker.repository.cloudera.com/cdsw/ml-runtime-jupyterlab-python3.8-standard:2021.09.1-b5

# Upgrade packages in the base image
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get -y install telnet wget build-essential cmake && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.2.tar.gz && \
  tar zxf openmpi-3.1.2.tar.gz && \
  cd openmpi-3.1.2 && \
  ./configure --enable-mpirun-prefix-by-default && \
  make -j && make install && \
  cd .. && rm -rf openmpi-3.1.2 openmpi-3.1.2.tar.gz

RUN  pip3 install tensorflow-cpu==2.3.4 && ldconfig && \
  HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip3 install --no-cache-dir horovod==0.21.3 && \
  pip3 cache purge

# Override Runtime label and environment variables metadata
ENV ML_RUNTIME_EDITION="Distributed Tensorflow" \
 ML_RUNTIME_SHORT_VERSION="1.0" \
 ML_RUNTIME_MAINTENANCE_VERSION="1"
ENV ML_RUNTIME_FULL_VERSION="$ML_RUNTIME_SHORT_VERSION.$ML_RUNTIME_MAINTENANCE_VERSION" ML_RUNTIME_DESCRIPTION="This runtime includes the package required to run distrubted model training for Tensorflow 2"

LABEL com.cloudera.ml.runtime.edition=$ML_RUNTIME_EDITION \
com.cloudera.ml.runtime.full-version=$ML_RUNTIME_FULL_VERSION \
com.cloudera.ml.runtime.short-version=$ML_RUNTIME_SHORT_VERSION \
com.cloudera.ml.runtime.maintenance-version=$ML_RUNTIME_MAINTENANCE_VERSION \
com.cloudera.ml.runtime.description=$ML_RUNTIME_DESCRIPTION

