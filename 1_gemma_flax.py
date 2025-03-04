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
### https://gemma-llm.readthedocs.io/en/latest/colab_sampling.html#
# -

# Common imports
import os, sys, time

REPO_NAME, BASE = 'getting-to-aha-with-tpus', './'
if not REPO_NAME in os.getcwd():
  # ! git clone https://github.com/mdda/getting-to-aha-with-tpus.git
  BASE = f'./{REPO_NAME}'
sys.path.append(BASE)  

import aha_library.platform
backend = aha_library.platform.detect()
uv_cmd, pip_install_jax = aha_library.platform.jax_pip_install_str(backend)
backend, pip_install_jax

# +
# # ! pip install -q git+https://github.com/google-deepmind/gemma.git

# Actually needed to specify these (latest) versions to kill looping resolution (on GCP VM)
# #! pip install gemma tensorflow[and-cuda] 'tfds-nightly==4.9.7.dev202502220044' 'google_cloud_resource_manager==1.14.1' 'grpcio_status==1.70.0'

## Started with bare venv = '~/flax_gemma/'
# #! uv pip install plotly treescope
# ! {uv_cmd} pip install etils msgpack absl-py rich tqdm
# ! {uv_cmd} pip install flax einops
# ! {uv_cmd} pip install kauldron
# #! {uv_cmd} pip install numpy
#"DONE"

# +
#import numpy as np
#def dummy_npwarn_decorator_factory():
#  def npwarn_decorator(x):
#    return x
#  return npwarn_decorator
#np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)
# -

#NOPE#! git clone -b gemma2-2b https://github.com/mdda/flax.git {repo_gemma_nnx_dir}/flax
if False:
  # ! git clone https://github.com/google-deepmind/gemma.git ./gdm-gemma

# +
import jax
import jax.numpy as jnp

# Gemma imports
sys.path.append(f"./gdm-gemma")
from gemma import gm
from gemma import peft  # Parameter fine-tuning utilities
sys.path.pop();

import treescope
# -

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
jax.default_backend()

# +
#from omegaconf import OmegaConf
#config = OmegaConf.load('./config.yaml')
#config.model.kaggle_id, config.model.weights_dir, config.model.ckpt_path

# +
#gm.ckpts.CheckpointPath.GEMMA2_2B_IT
## AttributeError: module 'orbax.checkpoint' has no attribute 'options' :: REINSTALLED EVERYTHING IN CLEAN venv
# <CheckpointPath.GEMMA2_2B_IT: 'gs://gemma-data/checkpoints/gemma2-2b-it/'>
# GEMMA2_2B_PT = Base model
# -

model = gm.nn.LoRAWrapper(
  rank=4,
  model=gm.nn.Gemma2_2B(tokens="batch.input"),
)

# +
token_ids = jnp.zeros((1, 256,), dtype=jnp.int32)  # Create the (batch_size, seq_length)

params = model.init(
  jax.random.key(0),    
  # This randomises everything - but we'll load pretrained params on top soon enough...
  token_ids,
)

params = params['params']  

# +
# Splits the params into non-LoRA and LoRA weights
original, lora = peft.split_params(params)

# Load the params from the checkpoint
chk_path_abs = os.path.abspath(f"{BASE}/{config.model.ckpt_path}")
#original = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA2_2B_IT, params=original)  # This is a cloud bucket address 
original = gm.ckpts.load_params(chk_path_abs, params=original)

# Merge the pretrained params back with LoRA
params = peft.merge_params(original, lora)
# -

#treescope.show(params)
jnp.set_printoptions(precision=4, floatmode='fixed')
params['final_norm']['scale'][0:200:20]

import flax
params_flat = flax.traverse_util.flatten_dict(params, sep='/')
for k in sorted(params_flat.keys()):
  if 'lora' in k: continue
  if 'layer_1' in k: break
  v = jnp.ravel(params_flat[k])[:5]
  print(f"{k:>60s} : {v}")  

# ### Have a look at a single token sample...

# +
tokenizer = gm.text.Gemma2Tokenizer()

prompt_txt = 'The capital of France,'

# +
# Encode the prompt
prompt = tokenizer.encode(prompt_txt, add_bos=True)  # /!\ Don't forget to add the BOS token
prompt = jnp.asarray(prompt)

# Run the model repeatedly : 'logits in parallel mode'
for _ in range(10):
  t0=time.time()
  out = model.apply(
    {'params': params},
    tokens=prompt,
    #return_last_only=True,  # Only return the last token outputs
  )
  print(f"{(time.time()-t0)*1000.:.2f}msec")
# 11160.85msec   == jit delay
# 337.03msec     
# 211.90msec     == settled down
# 211.92msec
# 212.23msec
# -

out.logits  # These are the 'correct outputs'

# Sample a token from the predicted logits
next_token = jax.random.categorical(
  jax.random.key(11),
  out.logits[-1]
)
tokenizer.decode(next_token)

tokenizer.plot_logits(out.logits[-1])

# ### Now let's try sampling some responses...

# https://gemma-llm.readthedocs.io/en/latest/api/gm/text/Sampler.html
# Greedy decoding is built-in : https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py#L170
sampler = gm.text.Sampler(
  model=model,
  params=params,
  tokenizer=tokenizer,
)
# Initialise this in its own cell : This makes it so that it only needs to jit once

for bs in range(1,4+1):
  t0=time.time()
  sampler.sample([prompt_txt,]*bs, max_new_tokens=30)
  print(f"{bs=} {(time.time()-t0)*1000.:.2f}msec")
# Each first-time jit takes + ~15 secs - cached thereafter
# bs=1 21022.31msec
# bs=2 25384.49msec
# bs=3 25199.83msec
# bs=4 24870.40msec
# bs=1 15332.65msec <-- bs=1 has slightly different behaviour...
# bs=2 7868.63msec
# bs=3 8069.74msec
# bs=4 8236.88msec

# ### Good features
#
# * Actually produces results, without mystery or crashing machine
# * Has LoRA built in
#
# ### Problems identified
#
# * Sampler is purely greedy : 
#   + https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py#L170
# * No sharding in Sampler (it's a TODO)

# +
# #%load_ext autoreload
# #%autoreload 2
# -

# https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py#L81
class MySampler(gm.text.Sampler):
  def __init__(self, sampler):
    #super().__init__()
    #self.orig = orig_sampler
    pass
  def test(self):
    print("Testing MySampler")


my_sampler = MySampler(sampler)

my_sampler.test()  # Works
#my_sampler.model # Exists
#my_sampler.cache_length
my_sampler.dtype





# ### Testing
# On Colab, add debug to:
# * `/usr/local/lib/python3.11/dist-packages/gemma/transformer.py`
#   + Line 317 : `jax.debug.print("embedder {v}", v=jnp.ravel(x)[:10],)`
# * `/usr/local/lib/python3.11/dist-packages/gemma/gm/nn/_transformer.py`
#   + Top : `import jax`
#   + Line 141 : `jax.debug.print("embedder {v}", v=jnp.ravel(x)[:10],)`
# * Then to :
#   + modules.Block (break after layer_0)
#   + layers.RMSNorm
# * JUST A MINUTE!
#   + Order of operations in gemma.nnx Block is completely wrong...
#   + Fixing it gets us back to normality!


