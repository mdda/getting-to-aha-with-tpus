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
## Follows https://flax.readthedocs.io/en/latest/nnx_basics.html

# ! uv pip install -U flax sentencepiece

from flax import nnx
import sentencepiece as spm

# +
## https://flax.readthedocs.io/en/latest/guides/gemma.html
# ! uv pip install jaxtyping kagglehub treescope

import kagglehub
# -

kagglehub.login()


VARIANT       = '2b' # @param ['2b', '2b-it', '7b', '7b-it'] {type:"string"}
weights_dir   = kagglehub.model_download(f'google/gemma-2/flax/gemma2-{VARIANT}')
weights_dir

ckpt_path     = f'{weights_dir}/{VARIANT}'
vocab_path    = f'{weights_dir}/tokenizer.model'
