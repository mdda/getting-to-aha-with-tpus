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
### https://flax.readthedocs.io/en/latest/guides/gemma.html

# +
# # ! pip install -q git+https://github.com/google-deepmind/gemma.git

# Actually needed to specify these (latest) versions to kill looping resolution (on GCP VM)
# #! pip install gemma tensorflow[and-cuda] 'tfds-nightly==4.9.7.dev202502220044' 'google_cloud_resource_manager==1.14.1' 'grpcio_status==1.70.0'

# Started with bare venv = '~/flax_gemma/'
# ! uv pip install gemma "jax[cuda12]" # "tensorflow[with-cuda]" 
# ! uv pip install plotly treescope
"DONE"

# +
# Common imports
import os
import jax
import jax.numpy as jnp

# Gemma imports
from gemma import gm
from gemma import peft  # Parameter fine-tuning utilities

import treescope
# -

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
jax.default_backend()

from omegaconf import OmegaConf
config = OmegaConf.load('./config.yaml')
config.model.kaggle_id, config.model.weights_dir, config.model.ckpt_path

# +
#gm.ckpts.CheckpointPath.GEMMA2_2B_IT
## AttributeError: module 'orbax.checkpoint' has no attribute 'options' :: REINSTALLED EVERYTHING IN CLEAN venv
# <CheckpointPath.GEMMA2_2B_IT: 'gs://gemma-data/checkpoints/gemma2-2b-it/'>

# +
tokenizer = gm.text.Gemma2Tokenizer()

model = gm.nn.LoRAWrapper(
  rank=4,
  model=gm.nn.Gemma2_2B(tokens="batch.input"),
)

# +
token_ids = jnp.zeros((1, 256,), dtype=jnp.int32)  # Create the (batch_size, seq_length)

params = model.init(
  jax.random.key(0),    # This randomises everything - but we'll load in on top soon enough...
  token_ids,
)

params = params['params']  

# +
# Splits the params into non-LoRA and LoRA weights
original, lora = peft.split_params(params)

# Load the params from the checkpoint
#original = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA2_2B_IT, params=original)  # This is a cloud bucket address 
original = gm.ckpts.load_params(os.path.abspath(config.model.ckpt_path), params=original)

# Merge the pretrained params back with LoRA
params = peft.merge_params(original, lora)
# -

prompt_txt = 'The capital of France, '

# +
# Encode the prompt
prompt = tokenizer.encode(prompt_txt, add_bos=True)  # /!\ Don't forget to add the BOS token
prompt = jnp.asarray(prompt)

# Run the model
out = model.apply(
  {'params': params},
  tokens=prompt,
  return_last_only=True,  # Only return the last layer outputs
)

# Sample a token from the predicted logits
next_token = jax.random.categorical(
  jax.random.key(11),
  out.logits
)
tokenizer.decode(next_token)
# -

tokenizer.plot_logits(out.logits)

# https://gemma-llm.readthedocs.io/en/latest/api/gm/text/Sampler.html
# Greedy decoding is built-in : https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py#L170
sampler = gm.text.Sampler(
  model=model,
  params=params,
  tokenizer=tokenizer,
)
# Initialise this in its own cell : This makes it so that it only needs to jit once

sampler.sample([prompt_txt, prompt_txt], max_new_tokens=30)

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


