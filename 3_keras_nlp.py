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

# Common imports
import os, sys, time
import numpy as np

# %load_ext autoreload
# %autoreload 2

REPO_NAME, BASE = 'getting-to-aha-with-tpus', './'
if not REPO_NAME in os.getcwd():
  # ! git clone https://github.com/mdda/getting-to-aha-with-tpus.git
  BASE = f'./{REPO_NAME}'
sys.path.append(BASE)
BASE

import aha_library
#aha_library.beep()

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
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"  # Already handled in platform.jax_pip_install_str

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
  keras.config.set_floatx("float16")    
  # WORKS! : Model apparently loads/runs in 16-bit format
  pass
if backend=='tpu':
  keras.config.set_floatx("bfloat16")  

# +
n_devices, batch_dim, model_dim = len(jax.devices()), "batch", "model"

device_mesh = keras.distribution.DeviceMesh(
  #(1, n_devices),   # model spread over devices, same data for all
  (n_devices, 1),   # model on every device, different data for each one
  [batch_dim, model_dim],
  devices=keras.distribution.list_devices(),
)
device_mesh

# +
layout_map = keras.distribution.LayoutMap(device_mesh)

model_name = "gemma2_2b_en"
#model_name = "gemma2_9b_en"

# Layout is appropriate for 'gemma2_9b_en' (given in example code)
# Weights that match 'token_embedding/embeddings' will be sharded model_parallel-wise across TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in attention layers
layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*attention_output/kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear/kernel"] = (model_dim, None)

model_name

# +
model_parallel_distribution = keras.distribution.ModelParallel(
  layout_map=layout_map,
  batch_dim_name=batch_dim,
)

keras.distribution.set_distribution(model_parallel_distribution)

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
  spec = variable.value.sharding.spec if n_devices>1 else variable.value.sharding # SingleDeviceSharding    
  print(f'{variable.path:<50}  {str(variable.shape):<14}  {str(spec)}')

# ## Add LoRA

# Enable LoRA for the model and set the LoRA rank to 16.
gemma_lm.backbone.enable_lora(rank=16) 
# This appears to make the sampling regenerate compiled code (seems reasonable)

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
gemma_lm.sampler

t0=time.time()
print(gemma_lm.generate("How can I plan a trip to Europe?", max_length=100))
print(f"{(time.time()-t0)*1000.:.2f}ms") # Includes jit? ... ~20secs

# Time-test the sampling...
t0=time.time()
print(gemma_lm.generate("How can I plan a trip to Europe?", max_length=100))
print(f"{(time.time()-t0)*1000.:.2f}ms") # T4 ~ 64 ms/tok (32-bit), 39 ms/tok (16-bit), TPU v5-1 ~ 7 ms/tok

aha_library.beep()

# ## dataset 

# +
max_prompt_length, max_completion_length = 256, 512

R1_STYLE_SYSTEM_PROMPT = """
A conversation between User and Assistant. The user poses a countdown task problem, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e.,
<reasoning> reasoning process here </reasoning>
<answer> answer here </answer>
""".strip()

# +
# Python generator? NO NEED - can do it in batches
# "Instruction:\n{instruction}\n\nResponse:\n{response}"
# -


import aha_library.dataset.countdown as dataset
dataset.generate_puzzle(seed=1, as_structure=True)  # Set the seed, show an example
# { numbers=' '.join(str(n) for n in sorted(numbers)), target=str(target), proof=expression, }

group_size = 8


def get_item(difficulty=3):
  if difficulty<4: # Pretty easy...
    return dataset.generate_puzzle(as_structure=True, n_small=3, n_large=1, target_min=10, target_max=500)
  # The following is difficulty 6 (i.e. human hardness)
  return dataset.generate_puzzle(as_structure=True, n_small=4, n_large=2, target_min=100, target_max=999)


TASK_SPECIFIC_INSTRUCTIONS = """
Welcome to the Countdown Numbers Game!
Your answer must combine some (or all) of the Contestant Numbers using only '+', '-', '*' and '/' to make the Target Number exactly.
""".strip()


def item_add_prompt(item):
  item['prompt'] = (  # This is Alpaca-style (for a base model)
    f"### Instruction:\n{R1_STYLE_SYSTEM_PROMPT}\n\n{TASK_SPECIFIC_INSTRUCTIONS}\n" +
    f"### Input:\nContestant Numbers: {item['numbers']}\nTarget Number: {item['target']}\n" +
    f"### Response:\n<reasoning>"
  )
  return
item = get_item()
item_add_prompt(item)
print(item['prompt'], item['proof'])


def multiply_item_by_n(item, n):
  return [ dict(item) for _ in range(n) ]
def get_generate_input(item_arr):
  return [
    item['prompt']
    for item in item_arr
  ]


# +
#get_generate_input( multiply_item_by_n(item, 4) )
# -

if False:
  # Test generation sizes...
  for n in [1,2,4,8,16,32,40,48,]:   # 64 fails
    item_group = multiply_item_by_n(item, n)
    prompts = get_generate_input(item_group)
    t0=time.time()
    gemma_lm.generate(prompts, max_length=max_completion_length)
    print(f"{n=:2d} : {(time.time()-t0)*1000.:.2f}ms - with compilation") 
    t0=time.time()
    responses = gemma_lm.generate(prompts, max_length=max_completion_length)
    print(f"{n=:2d} : {(time.time()-t0)*1000.:.2f}ms total = {(time.time()-t0)*1000./n:.2f}ms each - after jit")
    #print("\n---\n".join(responses))
#GCP T4:
#g= 1 : 28346.46ms - with compilation   g= 1 :  6872.51ms - after jit  
#g= 2 : 33250.36ms - with compilation   g= 2 : 13764.17ms - after jit
#g= 4 : 36556.85ms - with compilation   g= 4 : 15392.34ms total = 3848.09ms each - after jit
#g= 8 : 41973.12ms - with compilation   g= 8 : 17694.56ms total = 2211.82ms each - after jit
#g=16 : 52507.11ms - with compilation   g=16 : 23224.15ms total = 1451.51ms each - after jit
#g=32 : 75464.74ms - with compilation   g=32 : 35000.94ms total = 1093.78ms each - after jit
#g=40 : 87707.46ms - with compilation   g=40 : 40422.75ms total = 1010.57ms each - after jit
#g=48 : 98845.31ms - with compilation   g=48 : 46245.67ms total = 963.45ms each - after jit
#g=64 == OOM (on 16Gb T4)

# ### Reward Functions + Advantages
# * [Adapted from private Colab Notebook](grpo_qwen-0-5b_single_t4_countdown-mdda)

# +
import re, textwrap # Standard library

def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""
    #return aha_dataset.countdown.extract_solution(text) # This was from the Berkeley original

reward_func_pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n*<answer>(?:(?!</answer>).)*</answer>$"
def format_reward_func(item_arr, **kwargs) -> list[float]:
  """Reward function that checks whether each response has the correct format."""
  return [ 
    1.0 if bool(re.match(reward_func_pattern, item['response'].strip())) else 0.0 
    for item in item_arr 
  ]

# For fn spec, see: https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
def correctness_reward_func(item_arr) -> list[float]:
  """Reward function that checks whether each response has the correct answer."""
  correct_arr=[]
  for item in item_arr:
    extracted_response = extract_xml_answer(item['response'])
    item['extracted_response']=extracted_response
    # For each item, scoring the answer requires the 'target' and 'numbers' (to verify the solution is not cheating)
    ground_truth = dict(target=int(item['target']), numbers=[int(n) for n in item['numbers'].split(' ')])
    #print(idx, extracted_response, ground_truth)
    score = dataset.compute_score(
      f"Assistant:<answer>{extracted_response}</answer>", # re-fake response for standardised parsing
      ground_truth, format_score=0, score=1,
    ) #  , do_print=True
    correct_arr.append(score>0)
    
  if True:  # Output the item_arr[0] results
    item = item_arr[0]
    response_wrapped = '\n'.join('\n'.join(textwrap.wrap(t, width=80)) for t in item['response'].splitlines() )
    print(f"Question: {item['prompt']}\nTarget: {item['target']} with Proof: {item['proof']}\n"+
          f"Response: {response_wrapped}\nExtracted: {item['extracted_response']}")
    
  print(''.join('✅' if correct else '❌' for correct in correct_arr))
  return [2.0 if correct else 0.0 for correct in correct_arr]

#correctness_reward_func([[dict(content='last_user_prompt_line_here')],],
#                       [[dict(content='<answer>23-14</answer>')],],  # Fake response for parsing
#                        target=['9',], numbers=['14 23',], proof=["(23-14)",],)
correctness_reward_func([
  dict( prompt='PROMPT_TEXT', response='<answer>23-14</answer>', # Fake response for parsing
        target='9', numbers='14 23', proof="(23-14)", ),
])


# +
def item_add_group(item, group=-1):  # The 'group' is an ID - just need to be distinct for each one
  item['group'] = group
  
item_add_group(item, group=0)
# -

t0=time.time()
item_group = multiply_item_by_n(item, group_size)
prompts = get_generate_input(item_group)
responses = gemma_lm.generate(prompts, max_length=max_completion_length)
for idx, response in enumerate(responses):
  item_group[idx]['response'] = response
print(f"{group_size=:2d} : {(time.time()-t0)*1000.:.2f}ms total = {(time.time()-t0)*1000./group_size:.2f}ms each - after jit")
aha_library.beep()

reward_format      = format_reward_func(item_group)
reward_correctness = correctness_reward_func(item_group)
for item, r_fmt, r_cor in zip(item_group, reward_format, reward_correctness):
  item['reward'] = r_fmt + r_cor

# Group Advantage function (all items must be annotated with 'group' and 'reward' 
grouped_items=dict()
for item in item_group:
  group=str(item['group'])
  if group not in grouped_items:
    grouped_items[group]=[]
  grouped_items[group].append(item)
#print(f"{list(grouped_items.keys())=}")
for group, items in grouped_items.items():  # Now each group (whatever order) is in its own list
  rewards = np.array( [ item['reward'] for item in items ], dtype=np.float32 )
  mean = rewards.mean()
  std = rewards.std(mean=mean)
  advantages = (
    rewards*0.0 if std==0.0 else (rewards-mean)/std
  ).clip(0., rewards.shape[0])  
  #print(f"{rewards=}, {mean=}, {std=}\n{advantages=}")
  for idx in range(advantages.shape[0]):
    items[idx]['advantage'] = advantages[idx]
item_group[1]  # An element in the item_group - just to check the advantages have propagated

# ## New loss function
# * Based on: https://x.com/shxf0072/status/1892668791303373042
# * Observation (where A is the advantage) :
#   + grad_R = d_{ A*exp(log_m - log_ref) }
#   + ...  = A* d_(log_m}
#   + So   loss = A*log_prop_model has the same gradient...
#   + i.e. loss = advantage*log_softmax(logits)

# +
PAD_TOKEN  = gemma_lm.preprocessor.tokenizer.pad_token_id       # CONST
VOCAB_SIZE = gemma_lm.preprocessor.tokenizer.vocabulary_size()  # CONST

import jax

# eg: https://keras.io/examples/generative/midi_generation_with_transformer/
@keras.utils.register_keras_serializable()
def loss_log_softmax_logits(y_true, y_pred):
  """y_true is (batch, time)[token_idx.floatx], y_pred is (batch, time, logits)[floatx]"""
  #mask = ops.cast(ops.logical_not(ops.equal(y_true, CONFIG.token_pad)), "float32")
  #y_true = ops.one_hot(ops.cast(y_true_idx, "int32"), CONFIG.vocabulary_size)
  #return ops.categorical_crossentropy(y_true, y_pred, from_logits=True) * mask
  
  #jax.debug.print("y_true_idx.shape {v}", v=y_true_idx.shape)
  #jax.debug.print("y_true.dtype {v}", v=y_true.dtype)
  #y_true.shape (Array(8, dtype=int32), Array(512, dtype=int32))                             y_true.dtype float16
  #y_pred.shape (Array(8, dtype=int32), Array(512, dtype=int32), Array(256000, dtype=int32)) y_pred.dtype float16
  
  mask = keras.ops.cast(keras.ops.logical_not(keras.ops.equal(y_true, PAD_TOKEN)), y_pred.dtype)
  
  #y_true = keras.ops.one_hot(keras.ops.cast(y_true, "int32"), VOCAB_SIZE)  # TOO BIG
  #return -keras.ops.sum(y_probs*y_true, axis=-1) * mask
  #y_true_idx = keras.ops.cast(y_true, "int32")
  #mask.shape (Array(8, dtype=int32), Array(512, dtype=int32)) mask.dtype float16
  #y_true.shape (Array(8, dtype=int32), Array(512, dtype=int32), Array(256000, dtype=int32)) y_true.dtype float16
  
  # https://keras.io/api/ops/nn/
  y_probs  = keras.ops.log_softmax(y_pred, axis=-1)
  y_true_idx = keras.ops.cast(y_true, "int32")


  # TypeError: take_along_axis indices must be of integer type, got float16
  y_chosen = keras.ops.take_along_axis(y_probs, y_true_idx[..., None], axis=-1)  
  
  #y_probs.shape (Array(8, dtype=int32), Array(512, dtype=int32), Array(256000, dtype=int32)) y_probs.dtype float16
  #dot = keras.ops.dot(y_probs, y_true)
  #jax.debug.print("dot.shape {v}", v=dot.shape)
  #jax.debug.print("dot.dtype {v}", v=dot.dtype)
  
  # ValueError: Incompatible shapes for broadcasting: shapes=[(8, 512, 1), (8, 512)]
  return -(y_chosen * mask[..., None])  # Loss should be minimised at maximum accepted probability
  # NB: Should average over the token length?  (i.e. sum(mask, axis=-1) ?
  
  #return 8.
  # TypeError: broadcast_shapes got incompatible shapes for broadcasting: (8, 512, 256000), (1, 8, 512).
  #OR? https://keras.io/api/ops/numpy/#dot-function  

PAD_TOKEN, VOCAB_SIZE
# -



# ### Set up trainer / sampler

# +
gemma_lm.preprocessor.sequence_length = 512
# Can add sampler with other stuff : https://keras.io/keras_hub/api/base_classes/causal_lm/
gemma_lm.compile(
  #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),"
  loss=loss_log_softmax_logits,
  optimizer=keras.optimizers.Adam(learning_rate=5e-5),
  #weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
  weighted_metrics="auto", 
  sampler = keras_hub.samplers.RandomSampler(temperature=0.7),  # Also possible to add in here
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
# -

def get_train_input(item_arr): # (inputs=responses, sample_weights=advantages)
  return [ item['response'] for item in item_arr ], np.array([ item['advantage'] for item in item_arr ], )


# +
#gemma_lm.fit(data, epochs=1, batch_size=4)
# +
# https://keras.io/api/models/model_training_apis/
# (inputs, ..., sample_weights)
responses, advantages = get_train_input(item_group)
#responses, advantages

train_batch = 2
responses = responses[:train_batch]
advantages = advantages[:train_batch]
#responses = responses[train_batch:train_batch*2]
#advantages = advantages[train_batch:train_batch*2]

# +
# https://github.com/keras-team/keras/blob/v3.8.0/keras/src/backend/jax/trainer.py#L710
t0=time.time()

gemma_lm.train_on_batch(x=responses, sample_weight=advantages)

tms=(time.time()-t0)*1000.
print(f"{len(responses)=:2d} : {tms:.2f}ms total = {tms/len(responses):.2f}ms each - after jit")
# len(responses)= 2 : 68120.49ms total = 34060.24ms each
# len(responses)= 2 : 22723.84ms total = 11361.92ms each - after jit
# len(responses)= 2 : 676.67ms total = 338.34ms each - after jit
# No memory problem...

# len(responses)= 4 
# 2025-03-09 20:30:12.522939: W external/xla/xla/hlo/transforms/simplifiers/hlo_rematerialization.cc:3021] 
# Can't reduce memory use below 8.47GiB (9095344140 bytes) by rematerialization; 
#   only reduced to 9.43GiB (10128982370 bytes), down from 11.60GiB (12459174498 bytes) originally

# len(responses)= 4 
# 2025-03-09 20:23:31.950183: W external/xla/xla/hlo/transforms/simplifiers/hlo_rematerialization.cc:3021] 
# Can't reduce memory use below 8.47GiB (9095339276 bytes) by rematerialization; 
#   only reduced to 10.28GiB (11043464098 bytes), down from 13.71GiB (14724338162 bytes) originally
# -
aha_library.beep()


