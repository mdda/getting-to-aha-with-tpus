# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### https://flax.readthedocs.io/en/latest/guides/gemma.html

# +
import os, sys
import time

## Install 'uv' 
#sudo snap install astral-uv --classic
#uv venv flax_nnx
#. ./flax_nnx/bin/activate
# include some package to get jupyter up correctly
#uv pip install jupyterlab ipywidgets jupytext OmegaConf
# -

# %load_ext autoreload
# %autoreload 2

REPO_NAME, BASE = 'getting-to-aha-with-tpus', './'
if not REPO_NAME in os.getcwd():
  # ! git clone https://github.com/mdda/getting-to-aha-with-tpus.git
  BASE = f'./{REPO_NAME}'
sys.path.append(BASE)

import aha_library.platform
backend = aha_library.platform.detect()
uv_cmd, pip_install_jax = aha_library.platform.jax_pip_install_str(backend)
# This pulls in the correct JAX for the platform - likely needs updating, even if already in VM image
# ! {uv_cmd} {pip_install_jax}  
backend, pip_install_jax

import jax
jax.default_backend()



# +
#import subprocess
#
#try:
#  subprocess.check_output('nvidia-smi')
#  try:
#    # We must have a GPU in the machine : Let's see whether JAX is installed, and knows about it
#    import jax 
#    #assert 'cuda' in ','.join([str(d) for d in jax.devices()]).lower()
#    assert 'gpu' in jax.default_backend()    
#  except:    
# #    ! uv pip install -U "jax[cuda12]"
#    import jax
#  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"    
#except:
#  # We're not on a cuda machine - let's see whether we're on a TPU one
#  if 'TPU_ACCELERATOR_TYPE' in os.environ:
#    # This is essential - even raw Colab TPU machines may have outdated JAX
# #    ! pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#    import jax 
#    assert 'tpu' in jax.default_backend()
#  else:  # We are on a CPU machine
#    try:
#      import jax  # Plain cpu version expected here...
#      assert 'cpu' in jax.default_backend()
#    except:    
# #      ! uv pip install -U "jax"
#      import jax
#

import jax.numpy as jnp
# JAX will preallocate 75% of the total GPU memory when the first JAX operation is run. 
#   https://docs.jax.dev/en/latest/gpu_memory_allocation.html
jax.default_backend()

# +
#from omegaconf import OmegaConf
#
#config = OmegaConf.load(f'{BASE}/config.yaml')
#for extra in [f'{BASE}/config_secrets.yaml']:
#  if os.path.isfile(extra):
#    config = OmegaConf.merge(config, OmegaConf.load(extra))

import aha_library.config
config = aha_library.config.read(BASE)  
aha_library.config.load_kaggle_secrets(config) # sets up kaggle environment variables 
    
config.model.GEMMA_VARIANT, config.model.kaggle_id, config.model.kaggle_dir, config.model.weights_dir
# -

# ## Load JAX/Flax version of `gemma2-2b` weights

# +
from IPython.display import clear_output  # Makes the kaggle download less disgusting
## https://flax.readthedocs.io/en/latest/guides/gemma.html

weights_dir = config.model.weights_dir
if not os.path.isdir(weights_dir):   # Only prompt for download if there's nothing there...
  #os.environ['KAGGLE_USERNAME'] = config.kaggle.username
  #os.environ['KAGGLE_KEY'] = config.kaggle.key
  #from google.colab import userdata
  #for k in 'KAGGLE_USERNAME|KAGGLE_KEY'.split('|'):
  #  os.environ[k]=userdata.get(k)
  import kagglehub
  kagglehub.whoami()
  weights_dir = kagglehub.model_download(config.model.kaggle_id)
  print(f"Kaggle downloaded to {weights_dir}")  # Does this have a version number?
  # Now move the weights into the right place
  weights_dir = f"{config.model.kaggle_dir}/{config.model.kaggle_id}"
  os.makedirs(weights_dir, exist_ok=True)
  # ! mv ~/.cache/kagglehub/models/{config.model.kaggle_id}/* {weights_dir}/
  weights_dir = config.model.weights_dir
  #assert weights_dir == config.model.weights_dir
#weights_dir  # '/home/andrewsm/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b/1'
# ! echo {weights_dir} && ls -l {weights_dir}
# -


