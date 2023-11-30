#!/bin/bash
huggingface-cli download meta-llama/Llama-2-7b-hf
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
huggingface-cli download --repo-type dataset berkeley-nest/Nectar
huggingface-cli download --repo-type dataset nvidia/HelpSteer
