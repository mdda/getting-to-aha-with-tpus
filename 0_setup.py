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

# %load_ext autoreload
# %autoreload 2

# +
## Install 'uv' ?
# -

## Follows https://flax.readthedocs.io/en/latest/nnx_basics.html
for phase in "nothing-new-required installations-performed".split(' '):
  try:
    from flax import nnx
    import sentencepiece as spm
    break # This worked!
    # ?? cannot import name 'Key' from 'flax.typing' (/home/andrewsm/env311/lib/python3.11/site-packages/flax/typing.py)
  except Exception as e:
    print(type(e), e)
    # ! pip install --upgrade pip
    # ! pip install -U flax jaxtyping sentencepiece 
    # ! pip install kagglehub treescope
f"Installed with {phase}"

## https://flax.readthedocs.io/en/latest/guides/gemma.html
import kagglehub

kagglehub.login()


# +
from IPython.display import clear_output  # Makes the kaggle download less disgusting

GEMMA_VARIANT = 'gemma2-2b' # @param ['gemma2-2b', 'gemma2-2b-it', 'gemma2-7b', 'gemma2-7b-it'] {type:"string"}
weights_dir   = kagglehub.model_download(f'google/gemma-2/flax/{GEMMA_VARIANT}')
weights_dir  # '/home/andrewsm/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b/1'
# -

# ! ls -l '/home/andrewsm/.cache/kagglehub/models/google/gemma-2/flax/gemma2-2b/1'

ckpt_path     = f'{weights_dir}/{GEMMA_VARIANT}'
vocab_path    = f'{weights_dir}/tokenizer.model'
