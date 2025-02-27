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

# ### https://flax.readthedocs.io/en/latest/guides/gemma.html

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
    # ! uv pip install --no-deps -U flax
    # ! uv pip install jaxtyping sentencepiece 
    # ! uv pip install kagglehub plotly treescope
f"Installed with {phase}"

# +
from omegaconf import OmegaConf

config = OmegaConf.load('./config.yaml')
for extra in ['./config_secrets.yaml']:
  if os.path.isfile(extra):
    config = OmegaConf.merge(config, OmegaConf.load(extra))
    
config.model.GEMMA_VARIANT, config.model.kaggle_id, config.model.kaggle_dir, config.model.weights_dir

# +
from IPython.display import clear_output  # Makes the kaggle download less disgusting
## https://flax.readthedocs.io/en/latest/guides/gemma.html

weights_dir = config.model.weights_dir
if not os.path.isdir(weights_dir):   # Only prompt for download if there's nothing there...
  os.environ['KAGGLE_USERNAME'] = config.kaggle.username
  os.environ['KAGGLE_KEY'] = config.kaggle.key
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

# ## Get Gemma Library (converted to NNX by Google)
#
# * Code trickery from : https://flax.readthedocs.io/en/latest/guides/gemma.html

import sys
#repo_gemma_nnx_dir = config.nnx.repo_dir
#if not os.path.isdir(repo_gemma_nnx_dir):
#  os.makedirs(repo_gemma_nnx_dir, exist_ok=True)
#  # clone the `flax` repo into 'repo_gemma_nnx_dir'
#  # Then, append the `examples/gemma` folder to the path for loading the `gemma` modules.
# #  ! git clone https://github.com/google/flax.git {repo_gemma_nnx_dir}/flax

repo_gemma_nnx_dir = config.nnx.repo_dir
if not os.path.isdir(repo_gemma_nnx_dir):
  os.makedirs(repo_gemma_nnx_dir, exist_ok=True)
  # clone the `flax` fork into 'repo_gemma_nnx_dir'
  # Then, append the `examples/gemma` folder to the path for loading the `gemma` modules.
  # ! git clone -b gemma2-2b https://github.com/mdda/flax.git {repo_gemma_nnx_dir}/flax

sys.path.append(f"{config.nnx.repo_dir}/flax/examples/gemma")
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

config_gemma2_2b = transformer_lib.TransformerConfig.gemma2_2b()
config_gemma2_2b

params['transformer']['layer_0']['post_attention_norm']['scale'] # .keys()

#transformer = transformer_lib.Transformer.from_params(params)  # This is for v1 models
transformer = transformer_lib.Transformer.from_params(params, config_gemma2_2b)

# +
#nnx.display(transformer)
# -

jnp.set_printoptions(precision=4, floatmode='fixed')
transformer.final_norm.scale[0:200:20]  # This *proves* that the model has loaded the params

for k in """
embedder/input_embedding final_norm/scale 
layer_0/attn/attn_vec_einsum/w layer_0/attn/kv_einsum/w layer_0/attn/q_einsum/w 
layer_0/mlp/gating_einsum=layer_0/mlp/gate_proj/kernel/value
layer_0/mlp/linear=layer_0/mlp/down_proj/kernel/value
layer_0/post_attention_norm/scale=layer_0/post_attn_norm/scale
layer_0/post_ffw_norm/scale
layer_0/pre_attention_norm/scale
layer_0/pre_ffw_norm/scale
""".strip().split():
  if k.startswith('#'):
    print(f"{k:>60s} : SKIPPED")  
    continue
  if '=' in k:
    k=k.split('=')[-1]
    #print(k)
  o = transformer
  for a in k.split('/'):
    if 'layer_0' in a:
      o = getattr(o, 'layers')[0]
    else:
      o = getattr(o, a)
  v = jnp.ravel(o)[:5]
  print(f"{k:>60s} : {v}") 

# GDM gemma library : 2b-it 
# ``` 
#                                     embedder/input_embedding : [0.0351562 -0.0229492 0.081543 -0.0019455 0.0786133]
#                                             final_norm/scale : [2.32812 2.34375 2.28125 2.23438 2.07812]
#                               layer_0/attn/attn_vec_einsum/w : [0.0090332 0.0100708 0.0155029 0.0114136 0.00349426]
#                                     layer_0/attn/kv_einsum/w : [-0.0055542 -0.00469971 0.00686646 0.00354004 0.00970459]
#                                      layer_0/attn/q_einsum/w : [-0.00701904 -0.00222778 0.00172424 0.00372314 -0.00460815]
#                                    layer_0/mlp/gating_einsum : [0.0027771 -0.00335693 -0.00897217 -0.00811768 0.00405884]
#                                           layer_0/mlp/linear : [-0.000249863 0.00778198 0.0151978 0.00415039 -0.00415039]
#                            layer_0/post_attention_norm/scale : [-0.53125 -0.515625 -0.490234 -0.527344 -0.601562]
#                                  layer_0/post_ffw_norm/scale : [-0.229492 -0.189453 -0.194336 -0.220703 -0.196289]
#                             layer_0/pre_attention_norm/scale : [0.116699 0.134766 0.192383 0.185547 0.155273]
#                                   layer_0/pre_ffw_norm/scale : [0.227539 0.208008 0.208008 0.202148 0.135742]
# ```

# ```
# # Notes : 
# #  layer_0/mlp/linear=layer_0/mlp/down_proj/kernel/value = confirmed
# #  layer_0/mlp/gating_einsum=layer_0/mlp/gate_proj/kernel/value :
# #    the gating matrix is 'double size' in deepmind gemma2 implementation, but separate in flax.nnx
# #      https://github.com/google-deepmind/gemma/blob/main/gemma/modules.py#L266
# #      https://github.com/google/flax/blob/main/examples/gemma/modules.py#L258
# #  Dealt with in transformer.py:
# #      if 'gate_proj' in mapped_path:
# #        state[mapped_path].value = val[0]
# #        state[mapped_path[:-2] + ('up_proj', 'kernel')].value = val[1]  if 'gate_proj' in mapped_path:
# #        state[mapped_path].value = val[0]
# #        state[mapped_path[:-2] + ('up_proj', 'kernel')].value = val[1]
# ```

# ### Let's run a prompt through the transformer to get the logits

vocab = spm.SentencePieceProcessor()
vocab.Load(config.model.vocab_path);

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
# -

logits, _ = transformer(prompt, positions, cache=None, attention_mask=attn_mask)
logits.shape, logits # Seems to be a full list of logits for the input
# Sadly, they do not match the 'correct outputs' from flax_gemma _AT ALL_

for tok_logits in logits[0]: # Look at each token
  token_next = jnp.argmax(tok_logits)  # This is greedy
  print(f"{tok_logits.shape=} {token_next:6d} -> {vocab.id_to_piece(int(token_next))}")  

# Sample a token from the predicted logits
next_token = jax.random.categorical(
  jax.random.key(11),
  logits[0][-1]
)
vocab.id_to_piece(int(next_token))

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


