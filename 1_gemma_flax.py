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
### https://flax.readthedocs.io/en/latest/guides/gemma.html
# -

# # ! pip install -q git+https://github.com/google-deepmind/gemma.git
# ! pip install -q gemma

# +
# Common imports
import os
import jax
import jax.numpy as jnp

# Gemma imports
from gemma import gm
# -

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

config = OmegaConf.load('./config.yaml')
config.model.kaggle_id, config.model.weights_dir

gm.ckpts.CheckpointPath.GEMMA2_2B_IT

# +
tokenizer = gm.text.Gemma2Tokenizer()

model = gm.nn.Gemma2_2B()

#params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA2_2B_IT)
params = gm.ckpts.load_params(config.model.weights_dir)

# +
# Encode the prompt
prompt = tokenizer.encode('My name is', add_bos=True)  # /!\ Don't forget to add the BOS token
prompt = jnp.asarray(prompt)

# Run the model
out = model.apply(
  {'params': params},
  tokens=prompt,
  return_last_only=True,  # Only return the last layer outputs
)

# Sample a token from the predicted logits
next_token = jax.random.categorical(
  jax.random.key(1),
  out.logits
)
tokenizer.decode(next_token)
# -

tokenizer.plot_logits(out.logits)

# +
sampler = gm.text.Sampler(
    model=model,
    params=params,
    tokenizer=tokenizer,
)

sampler.sample('My name is', max_new_tokens=30)
# -

STOP

model_lora = gm.nn.LoRAWrapper(
    rank=4,
    #model=gm.nn.Gemma2_2B(tokens="batch.input"),
    model=model,
)

# https://github.com/google-deepmind/gemma/blob/main/gemma/peft/README.md#params-surgery
params_with_lora = 

# +
sampler = gm.text.Sampler(
    model=model_lora,
    params=params,
    tokenizer=tokenizer,
)

sampler.sample('My name is', max_new_tokens=30)
