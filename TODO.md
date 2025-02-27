
## Notes to self...

* Fork nnx into mdda 
  + create new branch : https://github.com/mdda/flax/tree/gemma2-2b
  + copy over changes on gcp
  + (check out my branch in config.nnx.tmp_dir)
  + make sure it works 
* Debug flax on colab
* High mem for gcp version
* Try on TPU


* Debug printing (even works in jitted code, though it may be out-of-order)
```python
      #if i==0:
      #  jax.debug.print("_layer0 after block {v}", v=jnp.ravel(x)[:4],)
      #  jax.debug.print("BREAK AFTER layer0")
      #  break # 
```