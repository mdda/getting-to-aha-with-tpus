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
### https://www.kaggle.com/code/matthewdwatson/gemma-2-tpu-fine-tuning/notebook
# -

# #! uv pip install -q -U keras-nlp tensorflow-text
# ! uv pip install -U keras-hub tensorflow-text
# ! uv pip install -U tensorflow-cpu

# Common imports
import os, sys, time

REPO_NAME, BASE = 'getting-to-aha-with-tpus', './'
if not REPO_NAME in os.getcwd():
  # ! git clone https://github.com/mdda/getting-to-aha-with-tpus.git
  BASE = f'./{REPO_NAME}'
BASE

import aha_library.platform
backend = aha_library.platform.detect()
pip_install = aha_library.platform.jax_pip_install_str(backend)
# ! uv {pip_install}  # This pulls in the correct JAX for the platform - likely needs updating, even if already in VM image
backend, pip_install

import jax
jax.devices()
# CUDA : [CudaDevice(id=0)]
# TPU : ...

# +
import os

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"  # Already handled in platform.jax_pip_install_str
# -

import keras
import keras_hub
# Takes a while

if backend=='gpu':
  #keras.mixed_precision.set_global_policy("mixed_float16") 
  # Doesn't seem to change anything...
  #keras.config.set_dtype_policy("mixed_float16") # https://keras.io/api/mixed_precision/policy/
  # Doesn't seem to change anything...
  #keras.mixed_precision.set_global_policy("mixed_bfloat16") # ... try again...
  # CAUSES : XlaRuntimeError: UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_BF16_BF16_F32
  pass

# +
n_devices, model_dim, batch_dim = len(jax.devices()), "model", "batch"

device_mesh_model_parallel = keras.distribution.DeviceMesh(
  (1, n_devices),   # model spread over devices, same data for all
  [batch_dim, model_dim],
  devices=keras.distribution.list_devices(),
)
device_mesh_model_parallel

# +
layout_map = keras.distribution.LayoutMap(device_mesh_model_parallel)

model_name = "gemma2_2b_en"
#model_name = "gemma2_9b_en"

# Weights that match 'token_embedding/embeddings' will be sharded on 8 TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in attention layers
layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*attention_output/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear/kernel"] = (model_dim, None)

model_name

# +
model_parallel = keras.distribution.ModelParallel(
  layout_map=layout_map,
  batch_dim_name=batch_dim,
)

keras.distribution.set_distribution(model_parallel)

# +
# Need to set up kaggle credentials...
from omegaconf import OmegaConf
config = OmegaConf.load(f'{BASE}/config.yaml')
for extra in [f'{BASE}/config_secrets.yaml']:
  if os.path.isfile(extra):
    config = OmegaConf.merge(config, OmegaConf.load(extra))
os.environ['KAGGLE_USERNAME'] = config.kaggle.username
os.environ['KAGGLE_KEY'] = config.kaggle.key

# https://keras.io/keras_hub/api/models/gemma/gemma_causal_lm/
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(model_name)  #  Download of ~5Gb, nice formatting

# +
#gemma_lm.quantize("float16")  # Try this... :: NOPE - expects int8...
#gemma_lm.quantize("float8")  # Ok then...  This attempts all layers...
# -

decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
  if n_devices>1:
    print(f'{variable.path:<48}  {str(variable.shape):<14}  {str(variable.value.sharding.spec)}') # 
  else:
    print(f'{variable.path:<48}  {str(variable.shape):<14}  {str(variable.value.sharding)}') # SingleDeviceSharding    

# ## Add LoRA

# Enable LoRA for the model and set the LoRA rank to 8.
gemma_lm.backbone.enable_lora(rank=8) 
# This appears to make the sampling regenerate code

# ## Inference test

# +
print(gemma_lm.generate("How can I plan a trip to Europe?", max_length=100))
# First generation is v time-consuming

# GPU (greedy sampler is default)
# I'm planning a trip to Europe in the summer of 2017. I'm looking for some advice on how to plan a trip like this. 
# I'm looking for a trip that is not too expensive, but also not too cheap. 
# I'm looking for a trip that is not too long, but also not too short. 
# I'm looking for a trip that is not too crowded, but also not too empty. 
# I'm looking for a trip that is not too hot, but also not too cold. 
# -



## Test Sampler too
gemma_lm.compile(
  #sampler = keras_hub.samplers.GreedySampler()
  #sampler = keras_hub.samplers.TopPSampler(p=0.1, k=1000)  # setting k reduces number of token_idx to sort over
  sampler = keras_hub.samplers.RandomSampler(temperature=0.7)
)

t0=time.time()
print(gemma_lm.generate("How can I plan a trip to Europe?", max_length=100))
print(f"{(time.time()-t0)*1000.:.2f}ms") # Includes jit?

# Time-test the sampling...
t0=time.time()
print(gemma_lm.generate("How can I plan a trip to Europe?", max_length=100))
print(f"{(time.time()-t0)*1000.:.2f}ms") # T4 ~ 64 ms/tok (32-bit), 91 ms/tok (16-bit?)



# ## Quick dataset 
# ####  Refillable?

# +
#import json
#data = []
#with open('/kaggle/input/databricks-dolly-15k/databricks-dolly-15k.jsonl') as file:
#    for line in file:
#        features = json.loads(line)
#        # Filter out examples with context, to keep it simple.
#        if features["context"]:
#            continue
#        # Format the entire example as a single string.
#        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
#        data.append(template.format(**features))
#
# Truncate our data to speed up training.
#data = data[:2500]
# +
# Python generator?
# "Instruction:\n{instruction}\n\nResponse:\n{response}"
# -


# ## Training test

gemma_lm.preprocessor.sequence_length = 512
gemma_lm.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(learning_rate=5e-5),
  weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  sampler = keras_hub.samplers.RandomSampler(temperature=0.7),  # Add this in too
)
gemma_lm.summary()
# GPU (standard loading):  -- changing to float16 above seems to make no difference
#   Total params: 2,620,199,168 (9.76 GB)
#   Trainable params: 5,857,280 (22.34 MB)
#   Non-trainable params: 2,614,341,888 (9.74 GB)

# +
#gemma_lm.fit(data, epochs=1, batch_size=4)
# -




