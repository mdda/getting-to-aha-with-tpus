
## Notes to self...

* DONE Fork `nnx` into mdda 
  + create new branch : https://github.com/mdda/flax/tree/gemma2-2b
  + copy over changes on gcp
  + (check out my branch in `config.nnx.tmp_dir`)
  + make sure it works 
  + Submit PR for basic gemma2-2b stuff

* Next ideas for `nnx gemma2`
  + Upgrade Sampler to allow for plugins
  + Fix 27B model's [attention normalisation factor](https://github.com/google-deepmind/gemma/blob/main/gemma/transformer.py#L235)


* DONE Debug flax on colab (GDM gemma = flax.linen)

* Try notebook on TPU Colab
  + Colab Stats
    - Usage rate: approximately 4.5 per hour
    - System RAM : 2.2 / 47.1 GB
    - Disk : 15.7 / 225.3 GB
    - TPU_ACCELERATOR_TYPE='v5e-1',
  + DONE Look at memory available = 48G?
  + DONE Integrate Colab findings into repo version of `0_setup.py`
  + DONE Find usable indicators for being a TPU machine (for auto-detection)
  + DONE Count of TPU cores = 1?
    -                       TPU_WORKER_ID='0'
    -                       COLAB_TPU_1VM='1'
    -                  TPU_SKIP_MDS_QUERY='1'
    -           TPU_CHIPS_PER_HOST_BOUNDS='2,2,1'
    -                     TPU_HOST_BOUNDS='1,1,1'
    -                TPU_ACCELERATOR_TYPE='v5e-1'
    -                TPU_WORKER_HOSTNAMES='localhost'

* High mem GCP instance for `nnx` version
  + DONE Is 64Gb enough? 
    - NO : `jit` itself consumes 60Gb of RAM
  + DONE Can jit be better located for lower memory consumption?
    - NO : That was just for Transformer logits forward pass
  + Add probabalistic SamplerPlugin idea
    - Why???

* DONE Write up +/- points of different directions in README

* DONE Get keras/JAX working on Colab TPU v5-1
  + NB: do not use `uv`, even though it *looks like* it works

* Add to keras/JAX code:
  - TEST : Add data parallel option / different DeviceMesh
    + https://keras.io/api/distribution/model_parallel/
    + more likely applicable for 2B models - we just want fast output, model fits in 16Gb RAM easily
  - Add dataset generation
    + "A Python generator function yielding (inputs, targets) or (inputs, targets, sample_weights)."
    + DONE: Sampling: [""]xN :: `generate`
    + Training: (inputs:[], targets:[], sample_weights: [] ) # sample_weights=(reward=advantage)
    + Consider : `train_on_batch` method (regular `.fit()` has a lot more machinery)
      - what happens to gradient accumulation? - seems to be taken care of by optimiser
      - For these simple functions, don't even need a generator, etc : Just a data tuple
  - Add code to sample a batch of n (divisible by 8) at once
    + DONE: How many samples can be trained at once in 16Gb device? (32 easily)
    + Critical for decisions about batching:
      * Group size = 16 seems like a good idea
      * Batch size = 4 groups? (from Qwen 0.5B example) 
  - Can keras do Gradient Accumulation?
    + https://keras.io/api/optimizers/adamw/ : `gradient_accumulation_steps`
    + useful when your batch size is very small
    + Learning rate schedules will look at "real" iterations value (optimizer steps)
  - Add new loss class : NegativeLogProb(use_logits=True)
    + Actually no need, `from_logits` looks like it is the gemma2 default output
      * for `(y_true, y_pred, from_logits=True)` loss input
      * ?? Need another input (?) in order to get group# into training step
        - NO (already accounted for in advantage calculation)
  - create loop of generate-score-train, etc
  - Test on GCP T4 (for a start)
  - Test on Colab TPU v5-1

* Test keras/JAX on Kaggle v3-8

* Test keras/JAX on 'regular' GCP TPU




* Debug printing (even works in jitted code, though it may be out-of-order)
```python
      #if i==0:
      #  jax.debug.print("_layer0 after block {v}", v=jnp.ravel(x)[:4],)
      #  jax.debug.print("BREAK AFTER layer0")
      #  break # 
```
