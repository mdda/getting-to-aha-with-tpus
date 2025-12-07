# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import jax

NUM_TPUS = len(jax.devices())

# +
# See : https://www.kaggle.com/code/marculera/supervised-fine-tuning-full
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true'
)
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['LIBTPU_INIT_ARGS'] = '--xla_enable_async_all_gather=true'

jax.config.update('jax_enable_x64', False)  # Use 32-bit for speed
jax.config.update('jax_default_matmul_precision', 'high')  # BF16 matmuls

# +
import functools, humanize
def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")
      
show_hbm_usage()    
# -

import kagglehub
#os.environ['KAGGLE_USERNAME']=""
#os.environ['KAGGLE_KEY']=""

KAGGLE_MODEL_HANDLE = "google/gemma-3/transformers/gemma-3-1b-it"

# +
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib

print(f"Model handle: {KAGGLE_MODEL_HANDLE}")

local_model_path = kagglehub.model_download(KAGGLE_MODEL_HANDLE)
print(f"✓ Model downloaded to: {local_model_path}")
# -

os.environ['KAGGLE_USERNAME']


