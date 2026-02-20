

# .condarc 
vim /root/.condarc

envs_dirs:
  - /root/autodl-tmp/conda

conda create -n AUT python=3.10.14 -y

modelscope download --model EleutherAI/pythia-2.8b README.md --local_dir ./pythia-2.8b
pip install ./flash_attn-2.1.1+cu121torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

export WANDB_MODE=disabled
export WANDB_DISABLED=true


conda activate AUT2
cd autodl-fs/2025autumn/Leaning/direct-preference-optimization-main
./batch.sh

pip install ./flash_attn-2.1.1+cu121torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


./batch.sh

