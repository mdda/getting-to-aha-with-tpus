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
import os, time
import asyncio, gc  # Avoid some jupyter async issues; Memory clean-up

import numpy as np  # On CPU : used for tokeniser stuff

import jax
import jax.numpy as jnp

NUM_TPUS = len(jax.devices())

CKPT_DIR = "~/content/ckpts/"

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

# https://www.kaggle.com/code/danielwycoff/dsa-cast-tunix-nolora-from-scratch
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


# +
from flax import nnx

from tunix.generate import tokenizer_adapter as tokenizer_lib
#from tunix.models.gemma import model as gemma_lib    # gemma2!
#from tunix.models.gemma import params as params_lib  # gemma2!
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params as params_lib
#from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import sampler as sampler_lib

import optax
from orbax import checkpoint as ocp
import qwix  # For LoRA
# -

# Random seeds
SEED = 42
#jax_key = jax.random.PRNGKey(SEED)
nnx_rng = nnx.Rngs(SEED)
#rng = np.random.default_rng(SEED)
#np.random.seed(SEED)
#random.seed(SEED)

# +
import functools, humanize
def hbm_usage(display=True):
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)
  mem_arr = []
  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    if display:
      print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")
    mem_arr.append(used)
  if display:
    print(f"{len(jax.live_arrays())} arrays are live in HBM")
  return mem_arr
      
hbm_usage()    
# -

# ## Download the Model

from dotenv import load_dotenv
if not load_dotenv(override=True):
  load_dotenv('./tpu_dotenv/dotenv', override=True)
os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'][-4:], 


#from tqdm import tqdm_notebook as tqdm
import tqdm
import tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # Monkey-patch before Kagglehub load (Does not seem effective...)

import kagglehub
#kagglehub.login()                # user interaction not required - have set os.environ using dotenv()

# +
# We should be using the flax downloads from Kaggle (not HF/Torch etc)
#?? KAGGLE_MODEL_HANDLE = "google/gemma-3/transformers/gemma-3-1b-it"

# https://www.kaggle.com/code/windmaple/grpo-demo-gemma2-2b
#KAGGLE_MODEL_HANDLE = "google/gemma-2/flax/gemma2-2b-it"  
#model_config = gemma_lib.ModelConfig.gemma2_2b()

KAGGLE_MODEL_HANDLE = "google/gemma-3/flax/gemma3-1b-it"  
model_config = gemma_lib.ModelConfig.gemma3_1b()
# -

kaggle_ckpt_path = kagglehub.model_download(KAGGLE_MODEL_HANDLE)
print(f"✓ Model downloaded to: {kaggle_ckpt_path}")

# +
# Special tags, system prompt and template
REASONING_START, REASONING_END = "<reasoning>", "</reasoning>"
ANSWER_START, ANSWER_END = "<answer>", "</answer>"

MAX_PROMPT_LENGTH = 256     # Just a guess
MAX_GENERATION_STEPS = 1024 # Stated in the Kaggle competition notes

SYSTEM_PROMPT = f"""
You always reason carefully before answering.

For each question:
- First, think through the problem step by step.
- Put all of your step-by-step reasoning strictly between {REASONING_START} and {REASONING_END}.
- If the result of reasoning is a number, then the final answer should be only that number.
- Put the final answer strictly between {ANSWER_START} and {ANSWER_END}.

You MUST include both blocks, in this order.
""".strip()

TEMPLATE = f"""
<start_of_turn>user
{{system_prompt}}

{{question}}<end_of_turn>
<start_of_turn>model
{REASONING_START}
""".strip()  # NB: {REASONING_START} added to TEMPLATE for reasoning model outputs...
# -

tokenizer = tokenizer_lib.Tokenizer(  # sentencepiece is default...
  tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
)
EOS_TOKENS = [tokenizer.eos_id()] # Use tokenizer EOS id
print(f"{EOS_TOKENS=}")
hbm_usage() # 91k (no storage used on TPU)

question_sample = "What is the highest prime below 42?"
tok_test = tokenizer.tokenize(
  TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question_sample), 
  add_eos=False)  # Returns a np.array of np.int32
tok_test.shape  # question_sample :: 120 toks

# ## Build the model

# +
# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
MESH_COUNTS = (1, 1)  # Default
if NUM_TPUS == 8:
  # in https://www.kaggle.com/code/danielwycoff/dsa-cast-tunix-nolora-from-scratch
  MESH_COUNTS = (1, 4)  # Spread across first 4 TPUs 
  #MESH_COUNTS = (1, 8) # Spread across all TPUs ?
  #MESH_COUNTS = (8, 1) # in https://www.kaggle.com/code/marculera/supervised-fine-tuning-full
mesh = jax.make_mesh(MESH_COUNTS, ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,)*2) # or AxisType.Explicit
mesh

# +
model_nnx = params_lib.create_model_from_checkpoint(
  os.path.join(kaggle_ckpt_path, "gemma3-1b-it"),
  model_config, 
  mesh = mesh,
)

# Sync before going to next cell (attempt)
# Iterate over every single array and force the host to wait until the device has finished writing it.
#jax.tree.map(lambda x: x.block_until_ready(), nnx.state(model_nnx))
await asyncio.sleep(0)  # This is required to ensure that we can move on to next cell cleanly
#gc.collect()

hbm_usage() # [2,000,511,488]


# +
# Explicitly delete the model - this does work!
#del model_nnx
#del model_rollout
#del sampler_rollout

#gc.collect()   #  This is apparently required to get rid of TPU memory allocated to 'del' variables
#hbm_usage() # 1.9Gb -> 1.7Mb
# -
# ### Model Loading and LoRA Application
# These two functions work together to load a base model from a checkpoint and apply a LoRA (Low-Rank Adaptation) layer to it.
#
# * `get_ref_model`: Loads the complete Gemma model from a specified checkpoint path. It uses JAX sharding to distribute the model parameters across multiple devices.
# * `get_lora_model`: Takes the base model and applies LoRA layers to it. It uses a LoraProvider to select specific layers (like attention and MLP layers) to be adapted. The resulting LoRA-infused model is then sharded and updated to ensure it's ready for distributed training.

def get_lora_model_qwix(base_model, mesh, rank=RANK, alpha=ALPHA):
  """
  This apparently makes a separate copy of the model
  rather than using the base_model weights by-reference
  """
  lora_provider = qwix.LoraProvider(
    module_path=(
      ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
      ".*attn_vec_einsum"
    ),
    rank=rank, alpha=alpha,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
    base_model, lora_provider, **model_input, rngs=nnx_rng,
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


# model_nnx exists
hbm_usage(), len(jax.live_arrays())


if True:  # The qwix version seems to copy the original weights
  model_lora = get_lora_model_qwix(model_nnx, mesh=mesh)
  #nnx.display(model_lora)  This does appear to have LoRA adapters in it
hbm_usage() # Now 3.8 GiB (with 2 1B models loaded)


del model_nnx
hbm_usage() # Now 2.1 GiB (with 1x 1B+LoRA model loaded)

model_rollout = model_lora



# +
def kv_cache_estimate(batch_size, steps_max=(MAX_PROMPT_LENGTH + MAX_GENERATION_STEPS + 32)):
  # https://notes.kvfrans.com/7-misc/rl-infra.html
  return (
    batch_size * steps_max *  # But actual allocation is not dependent on steps_max
    model_config.num_layers * model_config.num_kv_heads * model_config.head_dim
    * 2 # K+V
    * 2 # sizeof(bfloat16
  )
                      
def build_sampler(rollout_model, tokenizer, model_config):  # CACHE_SIZE based on MAX_GENERATION_STEPS
  """NB: Need to pass in the actual model to be used"""
  return sampler_lib.Sampler(
    transformer=rollout_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
      #cache_size  = MAX_PROMPT_LENGTH + MAX_GENERATION_STEPS + 256,
      cache_size  = MAX_PROMPT_LENGTH + MAX_GENERATION_STEPS + 32,
      num_layers  = model_config.num_layers,
      num_kv_heads= model_config.num_kv_heads,
      head_dim    = model_config.head_dim,
    ),
  )


# -

def generate_answers(question_arr, sampler, steps_max=MAX_GENERATION_STEPS, 
                     temperature=0.7, top_k=50, top_p=0.95, seed=None):
  batch = [
    TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q)
    for q in question_arr
  ]
  out = sampler(
    input_strings=batch,
    temperature=temperature, top_k=top_k, top_p=top_p,
    max_generation_steps=steps_max,
    eos_tokens=EOS_TOKENS,
    seed=seed, echo=False,
  )
  text_arr = out.text[:]  # Copy
  del out # Release structure
  gc.collect() #?
  return text_arr


# This is quick...
t0=time.time()
sampler_rollout = build_sampler(model_rollout, tokenizer, model_config)
print(f"{(time.time()-t0):.2f} seconds to build sampler")

# +
question_long = "List all the prime numbers less than 1 million"
batch_size, steps_max = 1, 1024  # Defaults

#steps_arr = [5, 64,128,150,200,250,256, 512, 800, 1024]
#batch_arr = [1, 8, 16, 64, 128, 200, ]  # Ok for raw model (failed after)
#batch_arr = [1, 8, 16, 64, 128, ]  # Ok for LoRA model (failed after)
#batch_arr = [16, 32, 60, 64, 68, 100, 124, 128, 132]  # Detail Ok for LoRA model (failed after)
batch_arr = [32, 38, 56,60,64,68,72, 96,100,104, 120,124,128,132,136]  # Detail Ok for LoRA model (failed after)
#batch_arr = [256, ]  # Fails with RESOURCE_EXHAUSTED
res_arr = []

for batch_size in batch_arr:
#for steps_max in steps_arr:
  print(f"{steps_max=} {batch_size=}")
  for trial in [0,1,2]:  # trial==0 may include jit delays : Can throw that result away
    t0=time.time()
    kv_est = kv_cache_estimate(batch_size)  #  This will be allocated+deallocated - but we need to add...
    # Each new value of steps_max incurs a jit compilation delay... (but they do get cached!)
    ans_arr = generate_answers( [question_long]*batch_size, sampler_rollout, steps_max=steps_max)
    #  ans_arr[0] = 'Prime number is a positive integer greater than 1 that  ...' (gets truncated)
    elapsed = time.time()-t0
    hbm = [ m+kv_est for m in hbm_usage(display=False)]
    print(f"  {trial=}, {batch_size=:3d}, {elapsed:8.3f}sec, {hbm=}")  
    if trial>0:
      res_arr.append( dict(trial=trial, batch_size=batch_size, steps_max=steps_max, 
                           elapsed=elapsed, hbm0=hbm[0],) )
# -

#ans_arr[0]
#nnx.display(sampler_rollout)
hbm_usage()

import pickle
#with open('steps-vary_bs1.pkl', 'wb') as f:
with open('bs-more-detail-128_steps1024_lora.pkl', 'wb') as f:
  pickle.dump(res_arr, f)

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
# -

res_df = pd.DataFrame(res_arr)
res_df['ms_per_token'] = res_df['elapsed']*1000. / (res_df['steps_max']*res_df['batch_size'])


def regplot(x="steps_max", y="elapsed", include_origin=True):
  fig, ax = plt.subplots(figsize=(12,4))
  sns.regplot(data=res_df, x=x, y=y, ax=ax, truncate=False )
  if include_origin:
    ax.set(xlim=(0, None), ylim=(0, None))  # Choose defaults for max
  plt.show()
#regplot(y="elapsed")  # When steps varied
regplot(x="batch_size", y="ms_per_token")

#regplot(x="steps_max", y="hbm0")
regplot(x="batch_size", y="hbm0")

# +
#checkpointer = ocp.StandardCheckpointer()
#_, state = nnx.split(gemma)
#checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
#checkpointer.wait_until_finished()
#show_hbm_usage()    
# -
# ## Speed check on getting logits from model

model_logits = model_lora

# +
batch_size, steps_max = 8, 1024

# Model input function
def generate_fake_inputs(batch_size, question, blah="just some random tokens that'll get repeated", 
                        steps_max=MAX_GENERATION_STEPS):
  tok_question = tokenizer.tokenize( 
    TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question), 
    add_eos=False)  # Returns a np.array of np.int32
  tok_answer = tokenizer.tokenize( blah, add_eos=False)
  n_answers = (steps_max-tok_question.shape[0])//tok_answer.shape[0] +1 # Round up
  tok_full = np.concat([tok_question, np.tile(tok_answer, n_answers)], axis=0) 
  tok_full = tok_full[:steps_max]  #  Truncate to be exactly steps_max long
  tok_batch = np.tile(tok_full, (batch_size, 1))
  return jnp.asarray( tok_batch )  # (batch_size, seq_len)


# +
import tunix.sft

input_fake = generate_fake_inputs(batch_size, question_sample, steps_max=256)  # [B, T]

pad_mask = input_fake != tokenizer.pad_id()  
positions = tunix.sft.utils.build_positions_from_mask(pad_mask)
attn_mask = tunix.sft.utils.make_causal_attn_mask(pad_mask)

input_fake.shape, positions.shape, attn_mask.shape
# ((8, 256), (8, 256), (8, 256, 256))
# -

# CHECK : is there a BOS token at the front?  == YES (actual 2 of them...)
#jnp.array([self.vocab.bos_id()] + input_ids, dtype=jnp.int32)
input_fake[0][:4], tokenizer.bos_id()

logits, _ = model_logits(input_fake, positions, cache=None, attention_mask=attn_mask)
logits.shape, logits.min(), logits.mean(), logits.max()  # Seems to be a full list of logits for the input

for idx, tok_logits in enumerate(logits[0]): # Look at each token
  token_idx = tokenizer.decode(int(input_fake[0][idx])).strip()
  pred_greedy = jnp.argmax(tok_logits)  # This is greedy
  token_next = tokenizer.decode(int(pred_greedy)).strip()
  #print(f"{tok_logits.shape=} {token_idx:20s} {pred_greedy:6d} -> {token_next:20s}")  

# +
# Actual speed test

batch_size, steps_max = 1, 1024  # Defaults
steps_arr = [1024, 1024*2]
batch_arr = [1,2,4,6,8,10,]  #  12 is an OOM

res_arr = []

for batch_size in batch_arr:
#for steps_max in steps_arr:
  print(f"{steps_max=} {batch_size=}")

  input_fake = generate_fake_inputs(batch_size, question_sample, steps_max=steps_max)  # [B, T]

  pad_mask = input_fake != tokenizer.pad_id()  
  positions = tunix.sft.utils.build_positions_from_mask(pad_mask)
  attn_mask = tunix.sft.utils.make_causal_attn_mask(pad_mask)    
   
  for trial in [0,1,2]:  # trial==0 may include jit delays : Can throw that result away
    t0=time.time()
    kv_est = kv_cache_estimate(batch_size, steps_max)  
    logits, _ = model_logits(input_fake, positions, cache=None, 
                             attention_mask=attn_mask, output_hidden_states=False)
    logits.block_until_ready()  # Maybe?
    elapsed = time.time()-t0
    hbm = [ m+kv_est*0 for m in hbm_usage(display=False)]
    print(f"  {trial=}, {batch_size=:3d}, {elapsed:8.3f}sec, {hbm=}")  
    if trial>0:
      res_arr.append( dict(trial=trial, batch_size=batch_size, steps_max=steps_max, 
                           elapsed=elapsed, hbm0=hbm[0],) )

# -

import pickle
#with open('steps-vary_bs1.pkl', 'wb') as f:
#with open('logits_bs_steps1024_lora.pkl', 'wb') as f:
with open('scanlogits_bs_steps1024_lora.pkl', 'wb') as f:
  pickle.dump(res_arr, f)

# ## Try a memory-efficient parallel forward pass

# +
from functools import partial

#@nnx.jit()
@jax.jit
def forward_to_prelogits_no_cache(self_model, tokens, positions, attention_mask):
  new_cache = None
  # Taken from tunix gemma3 code 
  #  https://github.com/google/tunix/blob/main/tunix/models/gemma3/model.py#L918-L938
  x = self_model.embedder.encode(tokens)
  for i, layer in enumerate(self_model.layers):
    layer_name = f'layer_{i}'
    #layer_cache = cache[layer_name] if cache else None
    with jax.named_scope(layer_name):
      layer_cache_discarded, x = layer(
          x,
          positions,
          None, #layer_cache,
          attention_mask,
      )
    #if cache is not None:
    #  new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

  return self_model.final_norm(x)  # 'x' is the pre-logits stage...
  #if output_hidden_states:
  #  self_model.sow(nnx.Intermediate, 'all_hidden_states', x)
  #logits = self_model.embedder.decode(x)

@partial(jax.jit, static_argnames=['k'])
def compute_chunked_top_k(hidden_states, embedding_matrix, k=128):
  """
  Computes Top-K logits without ever materializing the full [Batch, Seq, Vocab] tensor.
  
  Args:
      hidden_states: [Batch, Seq, Hidden_Dim] (Output from Teacher Transformer)
      embedding_matrix: [Vocab, Hidden_Dim] (The output head weights)
      k: int
      
  Returns:
      top_vals: [Batch, Seq, K]
      top_inds: [Batch, Seq, K]
  """
    
  # 1. We scan over the Sequence dimension (axis 1).
  # hidden_states needs to be transposed to [Seq, Batch, Hidden_Dim] for easier scanning
  hidden_states_T = jnp.swapaxes(hidden_states, 0, 1)

  def scan_step(carry, x_t):
    # x_t shape: [Batch, Hidden_Dim] (This is the hidden state for 1 timestep)
    
    # Compute logits ONLY for this timestep
    # [Batch, Hidden] @ [Hidden, Vocab] -> [Batch, Vocab]
    # This is 1024x smaller than the full sequence matrix
    logits_t = jnp.matmul(x_t, embedding_matrix.T) 
    
    # Extract Top-K immediately
    vals_t, inds_t = jax.lax.top_k(logits_t, k)
    
    # We don't need to carry anything, so return None
    return None, (vals_t, inds_t)

  # 2. Run the scan loop
  _, (top_vals_T, top_inds_T) = jax.lax.scan(scan_step, None, hidden_states_T)
  
  # 3. Swap axes back to [Batch, Seq, K]
  top_vals = jnp.swapaxes(top_vals_T, 0, 1)
  top_inds = jnp.swapaxes(top_inds_T, 0, 1)
  
  return top_vals, top_inds


# -





# +
# Actual speed test

batch_size, steps_max = 1, 1024  # Defaults
steps_arr = [1024, 1024*2]
#batch_arr = [1,2,4,6,8,10,]  #  All work!
batch_arr = [8,16,32, 38, 56,60,64,68,72, 96,100,104, 120,124,128,132,136]  # Detail Ok for LoRA model (failed after)

res_arr = []

for batch_size in batch_arr:
#for steps_max in steps_arr:
  print(f"{steps_max=} {batch_size=}")

  input_fake = generate_fake_inputs(batch_size, question_sample, steps_max=steps_max)  # [B, T]

  pad_mask = input_fake != tokenizer.pad_id()  
  positions = tunix.sft.utils.build_positions_from_mask(pad_mask)
  attn_mask = tunix.sft.utils.make_causal_attn_mask(pad_mask)    
   
  for trial in [0,1,2]:  # trial==0 may include jit delays : Can throw that result away
    t0=time.time()
    kv_est=0
    #kv_est = kv_cache_estimate(batch_size, steps_max)  

    #logits, _ = model_logits(input_fake, positions, cache=None, 
    #                         attention_mask=attn_mask, output_hidden_states=False)
    prelogits = forward_to_prelogits_no_cache(model_logits, input_fake, positions, attn_mask)
    top_vals, top_inds = compute_chunked_top_k(prelogits, model_logits.embedder.input_embedding, k=128)
    top_vals.block_until_ready()
    
    elapsed = time.time()-t0
    hbm = [ m+kv_est*0 for m in hbm_usage(display=False)]
    print(f"  {trial=}, {batch_size=:3d}, {elapsed:8.3f}sec, {hbm=}")  
    if trial>0:
      res_arr.append( dict(trial=trial, batch_size=batch_size, steps_max=steps_max, 
                           elapsed=elapsed, hbm0=hbm[0],) )

# -
input_fake.shape


nnx.display(prelogits)


