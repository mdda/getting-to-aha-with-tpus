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
import os, gc

import jax
import jax.numpy as jnp

NUM_TPUS = len(jax.devices())

INTERMEDIATE_CKPT_DIR = "~/content/intermediate_ckpt/"
CKPT_DIR = "~/content/ckpts/"
# -

from dotenv import load_dotenv
if not load_dotenv(override=True):
  load_dotenv('./tpu_dotenv/dotenv', override=True)
os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'][-4:], 


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
#from tunix.models.gemma3 import model as gemma_lib
##from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
#from tunix.models.gemma3 import params_lib
#from tunix.generate import tokenizer_adapter as tokenizer_lib

from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib

import optax
from orbax import checkpoint as ocp

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

from tqdm import tqdm_notebook as tqdm
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

params = params_lib.load_and_format_params(
  #os.path.join(kaggle_ckpt_path, "gemma2-2b-it")
  os.path.join(kaggle_ckpt_path, "gemma3-1b-it")
)

#gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
gemma = gemma_lib.Transformer.from_params(params, version="3-1b-it")

# +
checkpointer = ocp.StandardCheckpointer()
_, state = nnx.split(gemma)
checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
checkpointer.wait_until_finished()

show_hbm_usage()    

# +
# Delete the intermediate model to save memory.
del params
del gemma
del state
gc.collect()

show_hbm_usage()    

# +
# ====== Sharding ======
MESH_COUNTS = (1, 1)  # Default
if NUM_TPUS == 8:
  # in https://www.kaggle.com/code/danielwycoff/dsa-cast-tunix-nolora-from-scratch
  MESH_COUNTS = (1, 4)  # Spread across first 4 TPUs 
  #MESH_COUNTS = (1, 8) # Spread across all TPUs ?
  #MESH_COUNTS = (8, 1) # in https://www.kaggle.com/code/marculera/supervised-fine-tuning-full
MESH = [MESH_COUNTS, ("fsdp", "tp")]

# ====== LoRA ======
RANK = 64
ALPHA = 64.0
# -

# ### Model Loading and LoRA Application
# These two functions work together to load a base model from a checkpoint and apply a LoRA (Low-Rank Adaptation) layer to it.
#
# * `get_ref_model`: Loads the complete Gemma model from a specified checkpoint path. It uses JAX sharding to distribute the model parameters across multiple devices.
# * `get_lora_model`: Takes the base model and applies LoRA layers to it. It uses a LoraProvider to select specific layers (like attention and MLP layers) to be adapted. The resulting LoRA-infused model is then sharded and updated to ensure it's ready for distributed training.

tokenizer = tokenizer_lib.Tokenizer(
  tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
)
show_hbm_usage()    


def get_gemma_ref_model(ckpt_path, model_config):
  mesh = jax.make_mesh(*MESH)
  #model_config = gemma_lib.ModelConfig.gemma2_2b()
  # Here, 'abs_' -> "Abstract" meaning structures without assigned memory (yet)
  abs_gemma: nnx.Module = nnx.eval_shape( 
    # computes the shape/dtype of a function without any FLOPs
    lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
    lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
    abs_state,
    nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh, model_config


ref_model, mesh, model_config = get_gemma_ref_model(
  ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state"),
  model_config
)
show_hbm_usage()    


def get_lora_model(base_model, mesh, rank=RANK, alpha=ALPHA):
  lora_provider = qwix.LoraProvider(
    module_path=(
      ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
      ".*attn_vec_einsum"
    ),
    rank=rank,
    alpha=alpha,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model



lora_policy = get_lora_model(ref_model, mesh=mesh)
nnx.display(lora_policy)
show_hbm_usage()







def build_sampler(policy_model, tokenizer, model_config):
    return sampler_lib.Sampler(
        transformer=policy_model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )



def generate_answers(questions, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    if isinstance(questions, str):
        batch = [
            TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=questions),
        ]
    else:
        batch = [
            TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q)
            for q in questions
        ]
    out = sampler(
        input_strings=batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
        eos_tokens=EOS_TOKENS,
    )
    texts = out.text
    return texts[0] if isinstance(questions, str) else texts

