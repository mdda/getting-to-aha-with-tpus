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

# +
import os

## Install 'uv' ?
# -

# %load_ext autoreload
# %autoreload 2

# +
import subprocess

try:
  subprocess.check_output('nvidia-smi')
  try:
    # We must have a GPU in the machine : Let's see whether JAX is installed, and knows about it
    import jax 
    #assert 'cuda' in ','.join([str(d) for d in jax.devices()]).lower()
    assert 'gpu' in jax.extend.backend.get_backend().platform
  except:    
    # ! pip install -U "jax[cuda12]"
except:
  # We're not on a cuda machine - let's see whether we're on a TPU one
  try:
    import jax 
    YIKES - which one to install?
    assert 'tpu' in jax.extend.backend.get_backend().platform
  except:    
    print("Figure out what is special about a TPU machine without having jax installed already?")
    # #! pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pass
# -

## Follows https://flax.readthedocs.io/en/latest/nnx_basics.html
for phase in "nothing-new-required installations-performed".split(' '):
  try:
    from flax import nnx
    import sentencepiece as spm
    from omegaconf import OmegaConf
    break # This worked!
    # ?? cannot import name 'Key' from 'flax.typing' (/home/andrewsm/env311/lib/python3.11/site-packages/flax/typing.py)
  except Exception as e:
    print(type(e), e)
    # ! pip install --upgrade pip
    # ! pip install -U flax jaxtyping sentencepiece 
    # ! pip install kagglehub treescope
    # ! pip install OmegaConf
f"Installed with {phase}"

config = OmegaConf.load('./config.yaml')
#GEMMA_VARIANT = 'gemma2-2b' # @param ['gemma2-2b', 'gemma2-2b-it', 'gemma2-7b', 'gemma2-7b-it'] {type:"string"}
#weights_dir = '/home/andrewsm/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b/1'
#kaggle_dir = f'./models'
config.model.GEMMA_VARIANT, config.model.weights_dir

# +
## https://flax.readthedocs.io/en/latest/guides/gemma.html
from IPython.display import clear_output  # Makes the kaggle download less disgusting
import kagglehub

weights_dir = config.model.weights_dir
if not os.path.isdir(weights_dir):   # Only prompt for download if there's nothing there...
  kagglehub.login()
  os.makedirs(weights_dir, exist_ok=True)
  weights_dir = kagglehub.model_download(config.model.kaggle_model, path=kaggle_dir)
  assert weights_dir == config.model.weights_dir
#weights_dir  # '/home/andrewsm/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b/1'
# ! ls -l {weights_dir}
# -

# ## Get Gemma Library (converted to NNX by Google)
#
# * Code trickery from : https://flax.readthedocs.io/en/latest/guides/gemma.html

import sys
tmp_gemma_nnx_dir = config.nnx.tmp_dir
if not os.path.isdir(tmp_gemma_nnx_dir):
  os.makedirs(tmp_gemma_nnx_dir, exist_ok=True)
  # clone the `flax` repo into 'tmp_gemma_nnx_dir'
  # Then, append the `examples/gemma` folder to the path for loading the `gemma` modules.
  # ! git clone https://github.com/google/flax.git {tmp_gemma_nnx_dir}/flax

sys.path.append(f"{config.nnx.tmp_dir}/flax/examples/gemma")
import params as params_lib
import sampler as sampler_lib
import transformer as transformer_lib
sys.path.pop();

params = params_lib.load_and_format_params( os.path.abspath(config.model.ckpt_path) )
# NB: This is loaded on CPU : Nothing in GPU memory yet

vocab = spm.SentencePieceProcessor()
vocab.Load(config.model.vocab_path);

transformer = transformer_lib.Transformer.from_params(params)
nnx.display(transformer)


