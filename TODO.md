
## Notes to self...

* DONE Fork nnx into mdda 
  + create new branch : https://github.com/mdda/flax/tree/gemma2-2b
  + copy over changes on gcp
  + (check out my branch in config.nnx.tmp_dir)
  + make sure it works 
  + Submit PR for basic gemma2-2b stuff

* DONE Debug flax on colab (GDM gemma = flax.linen)

* Try on TPU Colab
  + Colab Stats
    - Usage rate: approximately 4.5 per hour
    - System RAM : 2.2 / 47.1 GB
    - Disk : 15.7 / 225.3 GB
    - TPU_ACCELERATOR_TYPE='v5e-1',
  + DONE Look at memory available = 48G?
  + DONE Integrate Colab findings into repo version of `0_setup.py`
  + DONE Find usable indicators for being a TPU machine (for auto-detection)
  + Count of TPU cores = 1?
    -                       TPU_WORKER_ID='0'
    -                       COLAB_TPU_1VM='1'
    -                  TPU_SKIP_MDS_QUERY='1'
    -           TPU_CHIPS_PER_HOST_BOUNDS='2,2,1'
    -                     TPU_HOST_BOUNDS='1,1,1'
    -                TPU_ACCELERATOR_TYPE='v5e-1'
    -                TPU_WORKER_HOSTNAMES='localhost'

* High mem GCP instance for `nnx` version
  + Is 64Gb enough? 
    - NO : `jit` itself consumes 60Gb of RAM
  + Can jit be better located for lower memory consumption?
    - NO : That was just for Transformer logits forward pass
  + Add probabalistic SamplerPlugin idea
    - Why???

* Write up +/- points of different directions in README


* Try on 'regular' TPU
  + Write up for Sprint


* Debug printing (even works in jitted code, though it may be out-of-order)
```python
      #if i==0:
      #  jax.debug.print("_layer0 after block {v}", v=jnp.ravel(x)[:4],)
      #  jax.debug.print("BREAK AFTER layer0")
      #  break # 
```
