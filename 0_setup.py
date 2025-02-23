# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: cache-notebooks//ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os

## Install 'uv' 
#sudo snap install astral-uv --classic
#uv venv flax_nnx
#. ./flax_nnx/bin/activate
# include some package to get jupyter up correctly
#uv pip install jupyterlab jupytext OmegaConf
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
    assert 'gpu' in jax.default_backend()
  except:    
    # ! uv pip install -U "jax[cuda12]"
    import jax
except:
  # We're not on a cuda machine - let's see whether we're on a TPU one
  try:
    import jax 
    YIKES - which one to install?
    assert 'tpu' in jax.default_backend()
  except:    
    print("Figure out what is special about a TPU machine without having jax installed already?")
    # #! pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    import jax

import jax.numpy as jnp
# JAX will preallocate 75% of the total GPU memory when the first JAX operation is run. 
#   https://docs.jax.dev/en/latest/gpu_memory_allocation.html
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
jax.default_backend()
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
    # ! uv pip install flax jaxtyping sentencepiece 
    # ! uv pip install kagglehub plotly treescope
f"Installed with {phase}"

# +
from omegaconf import OmegaConf

config = OmegaConf.load('./config.yaml')
config.model.GEMMA_VARIANT, config.model.kaggle_id, config.model.kaggle_dir, config.model.weights_dir

# +
## https://flax.readthedocs.io/en/latest/guides/gemma.html
from IPython.display import clear_output  # Makes the kaggle download less disgusting
import kagglehub

weights_dir = config.model.weights_dir
if not os.path.isdir(weights_dir):   # Only prompt for download if there's nothing there...
  #kagglehub.login()
  os.makedirs(config.model.kaggle_dir, exist_ok=True)
  weights_dir = kagglehub.model_download(config.model.kaggle_id, path=config.model.kaggle_dir)
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
#sys.path.pop();

abs_path = os.path.abspath(config.model.ckpt_path)
params = params_lib.load_and_format_params( abs_path )
# NB: This is loaded on CPU : Nothing in GPU memory yet

metadata = params_lib.load_metadata( abs_path )
[k for k in metadata.keys() if 'orbax' in k]
## https://github.com/google/flax/blob/main/examples/gemma/transformer.py#L60
#metadata['somewhere in orbax checkpoint']  # This was used to detect other v2 models...

# +
#params=None  # Try and reclaim CPU memory - seems to acheive no RAM reduction
#metadata=None
# -

vocab = spm.SentencePieceProcessor()
vocab.Load(config.model.vocab_path);

# +
### Test : Can jax store a bfloat16? : YES
#import jax.numpy as jnp
#a = jnp.bfloat16(3)
#a
# -

# Copied from             : https://github.com/google-deepmind/gemma/blob/main/gemma/transformer.py#L168
#   and modified to match : https://github.com/google/flax/blob/main/examples/gemma/transformer.py#L154
num_layers = _NUM_LAYERS_GEMMA2_2B = 26
#cache_size = None
config_gemma2_2b = transformer_lib.TransformerConfig(
        num_layers=num_layers, # _NUM_LAYERS_GEMMA2_2B,
        num_embed=256128,
        embed_dim=2304,
        hidden_dim=9216,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        final_logit_softcap=30.0,
        attention_types=(
            transformer_lib.modules.AttentionType.LOCAL_SLIDING,
            transformer_lib.modules.AttentionType.GLOBAL,
        )
        #* int(_NUM_LAYERS_GEMMA2_2B / 2),
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        #max_cache_length=cache_size,
        #query_pre_attn_norm=transformer_lib.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )
config_gemma2_2b

params['transformer']['layer_0']['post_attention_norm']['scale'] # .keys()

#transformer = transformer_lib.Transformer.from_params(params)  # This is for v1 models
transformer = transformer_lib.Transformer.from_params(params, config_gemma2_2b)

nnx.display(transformer)

# ### Let's run a prompt through the transformer to get the logits

prompt_txt = 'The capital of France,'

prompt = vocab.encode(prompt_txt, add_bos=True)  # /!\ Don't forget to add the BOS token
prompt = jnp.asarray([prompt])  # [List[int]] -> jnp.array[[]]
vocab.encode_as_pieces(prompt_txt), prompt

# +
prompt_len = prompt.shape[1]
input_mask = jnp.ones( prompt_len, dtype=jnp.bool)[None, :] # Allow all tokens

positions  = transformer_lib.build_positions_from_mask(input_mask)
attn_mask  = transformer_lib.make_causal_attn_mask(input_mask)
input_mask, positions, attn_mask

# +
#decoding_step = jnp.asarray([prompt_len], dtype=jnp.uint32)
#decoding_step = [prompt_len]
#decoding_step = 0
#decoding_step =jnp.arange(prompt_len, dtype=jnp.int32)
#attention_mask = sampler_lib._compute_attention_masks(decoding_step, prompt_len, input_mask)
#attention_mask
# -

logits, _ = transformer(prompt, positions, cache=None, attention_mask=attention_mask)
logits.shape, logits # Seems to be a full list of logits for the input

for tok_logits in logits[0]: # Look at each token
  token_next = jnp.argmax(tok_logits)
  print(f"{tok_logits.shape=} {token_next:6d} -> {vocab.id_to_piece(int(token_next))}")  

# ### Now let's try and sample some output

# Here, batch_size==1.  Having a different batch_size will trigger recompilation
input_batch = [
  #"\n# Python program for implementation of Bubble Sort\n\ndef bubbleSort(arr):",
  prompt_txt,
]

sampler = sampler_lib.Sampler(
  transformer=transformer,
  vocab=vocab,
)

import time
t0=time.time()
out_data = sampler(
  input_strings=input_batch,
  total_generation_steps=10,  # The number of steps performed when generating a response.
)
print(f"Overall : {time.time()-t0:.2f}sec")
# cache['v'].shape=(1, 1024, 4, 256)

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
  print()
  print(10*'#')


STOP


transformer=None


