FROM  tensorrt_llm/devel
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3-virtualenv
RUN python -m pip install --upgrade pip
RUN mkdir -p /code/tensorrt_llm && git clone https://github.com/NVIDIA/TensorRT-LLM.git /code/tensorrt_llm && \
	cd /code/tensorrt_llm && git submodule update --init --recursive && git lfs install && git lfs pull
RUN cd /code/tensorrt_llm && python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt && pip install build/*.whl
RUN pip install -U pip pynvml
RUN cd /code/tensorrt_llm/examples/quantization/ && \
	python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}') && \
	wget https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.5.0.tar.gz && \
	tar -xzf nvidia_ammo-0.5.0.tar.gz && \
	pip install nvidia_ammo-0.5.0/nvidia_ammo-0.5.0-cp$python_version-cp$python_version-linux_x86_64.whl && \
	pip install -r requirements.txt
RUN pip install -U pynvml dask dask-cuda distributed  cugraph cugraph-service-server cuml dask-cudf raft-dask
RUN pip list
RUN chown -R 1000 /code
