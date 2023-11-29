#!/bin/bash

#Install gh


type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)

type -p gh >/dev/null && echo Found gh || (curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y)


cd ${HOME}
[ -d venv ] && rm -rf ${HOME}/venv
python3 -m virtualenv venv
source ${HOME}/venv/bin/activate
pip install -U pip setuptools wheel
pip install --upgrade torch huggingface transformers transformers[torch] trl
pip list
