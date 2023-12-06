#!/bin/bash

docker run -u $(id -u) -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 tensorrt_llm/devel bash
