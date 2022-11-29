#! /usr/bin/env zsh

echo "Setting up your Mac for Pytorch"
echo "==============================="


echo "Installing PyTorch"
conda create --name="metal" python
conda activate metal
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install wandb tqdm

echo "Installing Huggingface Stack"
pip install transformers datasets