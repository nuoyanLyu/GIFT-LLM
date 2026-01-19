# Set up gift environment
We use python 3.12 and cuda 12.4.

```shell
conda create -n gift python=3.12
source activate gift
```
Then install ragen packages and pytorch.
```shell
cd GIFT-LLM
pip install -e .

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
Install other packages from `requirements.txt`.

```shell
pip install -r requirements.txt
```

To shorten the compilation time of flash-attn, we recommend to download the flash-attn wheel file from the official website https://github.com/Dao-AILab/flash-attention/releases. 
From our version, we use the flash-attn 2.7.4.post1, and the file name is `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`.

After download this, please install flash-attn through pip commands:
```shell
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

> Optional: You can install the flash-infer to boost the training. Similarly, we recomment to download the wheel file from the official website https://github.com/flashinfer-ai/flashinfer/releases. 
> From our version, we use the flash-infer 0.2.2.post1, and the file name is `flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl`.
Initialize and fetch submodule  VeRL.
```shell
git submodule update --init --recursive
```

> ⚠️ Important
> This project depends on a pinned commit of `verl@1e47e41`.
> Other versions of verl are not compatible with the current codebase.

We use `swanlab` to log the training process https://swanlab.cn/. Please register online and login through commands:
```shell
swanlab login
```
