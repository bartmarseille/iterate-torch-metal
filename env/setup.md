```commandline
cd projects/py
mkdir torch-m1-gpu
cd torch-m1-gpu
```
```commandline
touch environment.yml
```
edit `environment.yaml`
```commandline
name: torch-metal
channels:
  - conda-forge
  - defaults
dependencies:
    - python=3.9
    - pip>=19.0
    - jupyter
    - numpy
    - matplotlib
    - opencv
    - tqdm
    - wandb
    - pip:
        - --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
        - -c huggingface transformers datasets
```
```
conda env create
```