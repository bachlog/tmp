# wsl-pytorch

# Set-up PyTorch dev. env. for WSL Ubuntu on Windows 11, with Nvidia 1070Ti gpu.
- Ref: https://github.com/pytorch/pytorch/issues/73487

## Step 1. Install WSL2 Ubuntu on Windows
- Ref: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-wsl
- Command: `wsl --install`
- Note: If current WSL not work, or WSL 1, should uninstall by Windows Uninstaller and re-install WSL 2. Check WSL 2 on Windows cmd: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

## Step 2. Install CUDA Toolkit on WSL2 Ubuntu
- Do not install the default, and avoid 11.3
- Install CUDA Toolkit 11.7 as guided by Nvidia: https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
- Instructions:
  ```
  $ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  $ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  $ wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
  $ sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
  $ sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
  $ sudo apt-get update
  $ sudo apt-get -y install cuda
  ```
- Or can go with 11.5: https://developer.nvidia.com/cuda-11-5-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
- Verify: check with `nvidia-smi`

## Step 3. Install python env and PyTorch
- Example: install [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
- Create a minimal working env: `conda create -n py39 python=3.9`
- Ref: [pytorch](https://github.com/pytorch/pytorch/issues/73487)
- Instructions:
  ```
  pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
  ```
  Check it as follows:
  ```
  Python 3.9.17 (main, Jul  5 2023, 20:41:20)
  [GCC 11.2.0] :: Anaconda, Inc. on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import torch
  >>> torch.__version__
  '1.11.0+cu115'
  >>> torch.cuda.is_available()
  True
  >>> torch.tensor(1).cuda()
  tensor(1, device='cuda:0')
  ```
