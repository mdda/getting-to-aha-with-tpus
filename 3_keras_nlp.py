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
### https://www.kaggle.com/code/matthewdwatson/gemma-2-tpu-fine-tuning/notebook
# -

# Common imports
import os, sys, time

REPO_NAME, BASE = 'getting-to-aha-with-tpus', './'
if not REPO_NAME in os.getcwd():
  # ! git clone https://github.com/mdda/getting-to-aha-with-tpus.git
  BASE = f'./{REPO_NAME}'
sys.path.append(BASE)
BASE

import aha_library.platform
backend = aha_library.platform.detect()
uv_cmd, pip_install_jax = aha_library.platform.jax_pip_install_str(backend)
if backend != 'tpu':
  # ! {uv_cmd} {pip_install_jax}  # This pulls in the correct JAX for the platform - likely needs updating, even if already in VM image
backend, pip_install_jax

# This has to be done in this order, apparently...
# ! {uv_cmd} pip install -q -U tensorflow-cpu
# ! {uv_cmd} pip install -q -U keras-hub tensorflow-text
# ! {uv_cmd} pip install -q -U keras

# ! {uv_cmd} pip install -q OmegaConf

# +
# DEFAULTS on Colab TPU v5-1 :
# Using Python 3.11.11 environment at: /usr
#   jax==0.4.33 jaxlib==0.4.33
#   keras==3.8.0 tf-keras==2.18.0
#   numpy==1.26.4
#   tpu-info libtpu==2.18.0 libtpu-nightly==0.1.dev20240916+nightly
#   torch-xla
# -

# The Keras 3 distribution API is only implemented for the JAX backend for now
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate all TPU memory to minimize memory fragmentation and allocation overhead.
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"  # Already handled in platform.jax_pip_install_str

import jax
jax.devices()
# CUDA : [CudaDevice(id=0)]
# TPU : [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]

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
  keras.config.set_floatx("float16")    # TRY THIS OUT ON GPU!
  pass
if backend=='tpu':
  keras.config.set_floatx("bfloat16")  

# +
n_devices, batch_dim, model_dim = len(jax.devices()), "batch", "model"

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
import aha_library.config
config = aha_library.config.read(BASE)  
aha_library.config.load_kaggle_secrets(config) # sets up kaggle environment variables 

# https://keras.io/keras_hub/api/models/gemma/gemma_causal_lm/
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(model_name)  
#  Download of ~5Gb, nice formatting (implies that actual download is in 16-bit format...)

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
print(f"{(time.time()-t0)*1000.:.2f}ms") # Includes jit? ... ~20secs

# Time-test the sampling...
t0=time.time()
print(gemma_lm.generate("How can I plan a trip to Europe?", max_length=100))
print(f"{(time.time()-t0)*1000.:.2f}ms") # T4 ~ 64 ms/tok (32-bit), 39 ms/tok (16-bit), TPU v5-1 ~ 7 ms/tok



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

# +
gemma_lm.preprocessor.sequence_length = 512
# Can add sampler with other stuff : https://keras.io/keras_hub/api/base_classes/causal_lm/
gemma_lm.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(learning_rate=5e-5),
  weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  sampler = keras_hub.samplers.RandomSampler(temperature=0.7),  # Add this in too
)
gemma_lm.summary()

# GPU (standard loading):  -- changing to a 'policy' of float16 above seems to make no difference
#   Total params: 2,620,199,168 (9.76 GB)
#   Trainable params: 5,857,280 (22.34 MB)
#   Non-trainable params: 2,614,341,888 (9.74 GB)

# GPU (with .set_floatx() ) -- loads, and runs in 16-bit everywhere ! 
#   Total params: 2,620,199,168 (4.88 GB)
#   Trainable params: 5,857,280 (11.17 MB)
#   Non-trainable params: 2,614,341,888 (4.87 GB)

# TPU v5-1 (JAX backend)
#    Total params: 2,620,199,168 (4.88 GB)
#    Trainable params: 5,857,280 (11.17 MB)
#    Non-trainable params: 2,614,341,888 (4.87 GB)

# +
#gemma_lm.fit(data, epochs=1, batch_size=4)
# -




