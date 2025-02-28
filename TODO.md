
## Notes to self...

* Fork nnx into mdda 
  + DONE create new branch : https://github.com/mdda/flax/tree/gemma2-2b
  + DONE copy over changes on gcp
  + DONE (check out my branch in config.nnx.tmp_dir)
  + DONE make sure it works 
  + DONE Submit PR for basic gemma2-2b stuff

* DONE Debug flax on colab
* Try on TPU Colab
  + DONE Find usable indicators for being a TPU machine (for auto-detection)
  + DONE Look at memory available = 48G?
  + Integrate Colab findings into repo version of `0_setup.py`
  + Count of TPU cores = 1?
* Try on 'regular' TPU
  + Write up for Sprint
* High mem for gcp version
  + Is 64Gb enough?
  + Can jit be better located for lower memory consumption?
  + Add probabalistic SamplerPlugin idea


* Debug printing (even works in jitted code, though it may be out-of-order)
```python
      #if i==0:
      #  jax.debug.print("_layer0 after block {v}", v=jnp.ravel(x)[:4],)
      #  jax.debug.print("BREAK AFTER layer0")
      #  break # 
```
