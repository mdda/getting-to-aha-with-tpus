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
## Follows https://flax.readthedocs.io/en/latest/nnx_basics.html

# ! uv pip install -U flax
# -

from flax import nnx
import jax
import jax.numpy as jnp


# %load_ext autoreload
# %autoreload 2

class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b


# +
model = Linear(2, 5, rngs=nnx.Rngs(params=0))
y = model(x=jnp.ones((1, 2)))

print(y)
nnx.display(model)


# +
class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
    return self.linear2(x)

model = MLP(2, 16, 5, rngs=nnx.Rngs(0))

y = model(x=jnp.ones((3, 2)))

nnx.display(model)


# -

# ## Perform model surgery

# +
class LoraParam(nnx.Param): pass

class LoraLinear(nnx.Module):
  def __init__(self, linear: Linear, rank: int, rngs: nnx.Rngs):
    self.linear_frozen = linear  # Really want to 'hide' this from the optimiser...
    #self.linear_frozen.tag='frozen'  # But it's the child Params that need this tag!
    self.A = LoraParam(jax.random.normal(rngs(), (linear.din, rank)))
    self.B = LoraParam(jax.random.normal(rngs(), (rank, linear.dout)))

  def __call__(self, x: jax.Array):
    return self.linear_frozen(x) + x @ self.A @ self.B

rngs = nnx.Rngs(0)
model = MLP(2, 32, 5, rngs=rngs)

# Model surgery.
model.linear1 = LoraLinear(model.linear1, 4, rngs=rngs)
model.linear2 = LoraLinear(model.linear2, 4, rngs=rngs)

y = model(x=jnp.ones((3, 2)))

nnx.display(model)

# +
import optax
#optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing(?)
#optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param) # Default (i.e. same as above)

# See: https://github.com/google/flax/issues/4167#issuecomment-2324245208
#optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=LoraParam) # Just the Lora params
# But this would ignore the non-LoRA layers...  (which may be a reasonable thing to do...)

# filter to unselect Params that are inside a '*_frozen' ... or...
# See: https://flax.readthedocs.io/en/latest/guides/filters_guide.html#the-filter-dsl
non_frozen_params = nnx.All(nnx.Param, nnx.Not(nnx.WithTag('frozen')))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=non_frozen_params) # 

optimizer


# -
# ### Test nested jit

# +
@jax.jit
def f(x):
  x=x*2
  x=x+5
  return x
  
@jax.jit
def g(x):
  x=x-5
  x=x/2
  return x

@jax.jit
def h(x):
  x=f(x)
  x=g(x)
  return x

print(jax.make_jaxpr(h)(7.))
# -

print(jax.jit(h).lower(7.).compile().as_text())



# ### Quick textwrap for long outputs 
# #### Respect existing line-breaks

# +
import textwrap

txt="This is very long "*8 + "\n" + "And this is the second line "*10 

print('\n---\n'.join('\n'.join(textwrap.wrap(t, width=120)) for t in txt.splitlines() ))
# -


len("John removes 6 pink hard hats, which is ( 4 times 2 = 8 ) green hard hats. Thus, the remaining hard hat counts are:")

# ### Test countdown generation


#import sys
#sys.path.append(f"./getting-to-aha-with-tpus")
import aha_dataset.countdown
#sys.path.pop();

aha_dataset.countdown.generator.generate_puzzle(seed=1)

aha_dataset.countdown.generator.generate_puzzle(as_structure=True)

# ### Test Keras/JAX indexing

import os
os.environ["KERAS_BACKEND"] = "jax"

# #! uv pip install keras
import keras
import numpy as np

x_np = np.array([
  [
    [100, 200, 300, 400, 500], 
    [222, 233, 321, 405, 502], 
    [101, 222, 123, 432, 543], 
  ],
  [
    [600, 700, 600, 555, 987], 
    [722, 833, 721, 435, 876], 
    [800, 900, 900, 640, 765], 
  ],  
])
x_keras = keras.ops.array(x_np)
x_keras.shape

idx_np=np.array([
  [ 1,3,0 ],
  [ 3,2,4 ],
])
idx_keras = keras.ops.array(idx_np)

# +
#x_keras[..., idx_keras]  # Nope, this indexing is broadcast

# https://keras.io/api/ops/numpy/#take_along_axis-function
keras.ops.take_along_axis(x_keras, idx_keras[..., None], axis=-1)  # Seems to work!
# -
# Other ideas (DID NOT WORK OUT)
x_2d = x_keras.reshape( (-1, x_keras.shape[-1]) )
idx_1d = idx_keras.flatten()
x_2d, idx_1d


#x_selected = x_2d[:, idx_1d]  # NOPE
x_selected = keras.ops.numpy.take(x_2d, idx_1d, axis=-1) # NOPE
x_selected


