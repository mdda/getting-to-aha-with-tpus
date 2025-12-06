
## JAX Resources

### kaggle-tunix

More stuff appearing in here...


### JAX (generic)

* [Flow Matching in 50 lines of JAX](https://x.com/cgarciae88/status/1867340873136038293)
  + Cristian Garcia at DeepMind OS : @cgarciae88

* Google Docs: 
  + [MNIST example (JAX.nnx) with training loop](https://flax.readthedocs.io/en/latest/mnist_tutorial.html)
  + [Fine-tuning `gemma-2b-it` using JAX and Flax](https://ai.google.dev/gemma/docs/jax_finetune) (seems like 'pure JAX')
    - Colab-Pro A100 GPU
      + Could also use Kaggle's (free) TPU v3-8  
      + Colab (free) TPU v2 is insufficient
        * Even though `optimizer = optax.sgd(training_cfg.learning_rate)`
    - The gemma library was written with:
      + JAX, Flax, nnx(?)
      + Orbax (a JAX-based library for training utilities like checkpointing), and 
      + SentencePiece (a tokenizer/detokenizer library).
    - More [documentation about packages](https://ai.google.dev/gemma/docs/jax_finetune#learn_more)
    - Good example ([Gemma v1](https://github.com/google-deepmind/gemma), though), since : 
      + `loss_fn` gets redefined
      + seems to inherit from JAX gemma (so has a sampler.sample() and a train() available)
        * `from gemma import transformer as transformer_lib`
      + has training loop too 

* [Fine-Tuning Gemma for RAG with JAX+JORA](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Finetune_with_JORA.ipynb)
  + Colab non-free GPU == A100
  + Can `model.push_to_hub("my-gemma-finetuned-model")` after training LoRA
    - `from jora.hf.__main__ import lorize_huggingface`
  + Leverages JAX's JIT compilation and tensor-sharding capabilities
  + [JORA: JAX Tensor-Parallel LoRA Library for Retrieval Augmented Fine-Tuning](https://arxiv.org/abs/2403.11366)
    - [GitHub repo](https://github.com/aniquetahir/JORA)
    - "Update (8-Apr-2024): JORA now supports Google's Gemma models"

* [Optax may have gradient_accumulation built in](https://github.com/google-deepmind/optax/issues/320)
  - [Optax documentation example (JAX/linen)](https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html)



### Gemma Models

* [Using pretrained Gemma for inference with Flax NNX](https://flax.readthedocs.io/en/latest/guides/gemma.html) **
  + Gemma1 : 2.6B + 7B
  + Gemma2 : 2B + 9B + 27B  (2, 8, 13 Trillion tokens training, respectively)
  + CodeGemma : 2B + 7B
  + [Gemma formatting and system instructions](https://ai.google.dev/gemma/docs/formatting)
  + Flax NNX `gemma.transformer.TransformerConfig.from_params` function

* [Getting Started with RecurrentGemma and Flax ()](https://www.kaggle.com/code/philculliton/getting-started-with-recurrentgemma-and-flax/notebook)
  + RecurrentGemma : 2B
    - Griffin... : [Fine-tuning the 2B Griffin model with Flax.nn](https://www.kaggle.com/code/philculliton/fine-tuning-the-2b-griffin-model-with-flax)
  + [Inference with RecurrentGemma using JAX and Flax.nn](https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_inference)
  + [Fine-tuning RecurrentGemma using JAX and Flax.nn](https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_finetune)
    - `model = recurrentgemma.Griffin(config)`
  + [flax/examples/gemma/](https://github.com/google/flax/tree/main/examples/gemma) 
    - code seems to include `nnx` and `nn`
  

### Keras (JAX backend)

* Kaggle [Gemma 2 TPU Fine-tuning](https://www.kaggle.com/code/matthewdwatson/gemma-2-tpu-fine-tuning)
  + Uses Keras and includes : 
    + sharding :`layout_map = keras.distribution.LayoutMap(device_mesh)`
    + LoRA : `gemma_lm.backbone.enable_lora(rank=8)`
  + Total params: 9,270,779,392 (34.54 GB) with Trainable params: 29,073,408 (110.91 MB)
* Kaggle [Distributed tuning with Gemma using Keras¶](https://www.kaggle.com/code/nilaychauhan/keras-gemma-distributed-finetuning-and-inference)
* OLD : [Make a custom loss function in keras](https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras)

* [`import keras_nlp` LoRA tuning of Gemma](https://ai.google.dev/gemma/docs/lora_tuning)


### LoRA for JAX

* [Unlocking the Power of LoRA: Streamlined NLP with EasyDeL in JAX](https:/medium.com/@erfanzare810/unlocking-the-power-of-lora-streamlined-nlp-with-easydel-in-jax-904896c10cb4)
  + Uses [EasyDeL](https://github.com/erfanzar/EasyDeL)
    - [EasyDeL Documentation](https://easydel.readthedocs.io/en/latest/)
    - "Accelerate, Optimize performance with streamlined training and serving options with JAX"
    - "Built on modern Flax NNX, it provides convenient and effective solutions for training and serving Flax/JAX models on TPU/GPU at scale"
  + Has `load_in_8bit`, `class DeepseekV3Attention(FlaxAttentionModule):`
  + [LoRA example for SFT `Mistral-7B-Instruct` on Kaggle TPUs](https://www.kaggle.com/code/citifer/easydel-causal-language-model-trainer-example)
  + [Training example in documentation](https://easydel.readthedocs.io/en/latest/supervised-fine-tuning-trainer-example.html)

* [LoRA - NNX LoRA classes](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/lora.html)
* [NNX parameter surgery](https://flax.readthedocs.io/en/latest/why.html#model-surgery)
  + [Model Surgery (less informative)](https://flax.readthedocs.io/en/latest/guides/surgery.html)
  + [Toy Example Code](https://github.com/google/flax/blob/main/examples/nnx_toy_examples/09_parameter_surgery.py)
* [`nnx.Optimizer` state can be filtered](https://github.com/google/flax/blob/main/flax/nnx/training/optimizer.py#L183)
  + [But the LoRA class already declares the right `lora_param_type` ~annotations](https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/lora.html#LoRA)

* [`lorax`](https://github.com/davisyoshida/lorax/blob/master/examples/huggingface_gpt2.py)
  - [May not be the best implementation](https://github.com/jax-ml/jax/discussions/16588)



## TPU training

* [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
  + 2025-02-01 - DeepMind : Highly recommended

* TPU memory (per core)
  + TPU v2 :  8 GB, TPU v3 : 16 GB
  + TPU v4 : 32 GB
  + TPUv5e : 16 GB, TPUv5p : 95 GB
  + [Cloud TPU performance guide](https://cloud.google.com/tpu/docs/performance-guide)
  + To minimize memory overhead and maximize computational efficiency, one of the following must be true:
    - The total batch size should be a multiple of 64 (8 per TPU core), and feature dimension sizes should be a multiple of 128.
    - The total batch size should be a multiple of 1024 (128 per TPU core), and feature dimension sizes should be a multiple of 8.
  + The TPU *Node* architecture is being deprecated. 
    - TPU v2 and v3 are the only TPU versions that still support the TPU Node architecture
    - TPU v4 and newer only support the TPU VM architecture

* [Kaggle TPUs](https://ai.google.dev/gemma/docs/jax_finetune):
  + 30 hours per week of TPU-v3-8 and 
  + up to 3 hours at a time in a single session
  + PU profiling information is now reported in the Notebooks UI

* Pricing per core + Quotas: (# of premptible cores, zone)
  + TPU v2  : \$1.31 : 32+, EU+US, 16 asia-east1
  + TPU v3  : \$2.20 : 32+, EU+US
  + TPU v4  : \$3.22 : 'unlimited', everywhere
  + TPU v5e : \$1.56 : 32, EU+US+Asia
  + TPU v5p : \$4.20 :  'unlimited', EU+US+Asia ("talk to sales")
  + TPU v6e : \$2.97? : 'unlimited', EU+US+Asia ("preview")


* [Felafax - tune LLaMa3.1 on Google Cloud TPUs for 30\% lower cost and scale seamlessly!](https://github.com/felafax/felafax?tab=readme-ov-file#-finetune-for-free)
  - Full range of Llama models in JAX; Has LoRA implementation
  - Seems to 'take care of' spinning up everything (i.e. may be over-kill)
  - Reddit post by creators : [Tune LlaMa3.1 (written in JAX) for free on Google Colab TPU](https://www.reddit.com/r/LocalLLaMA/comments/1fj9hea/tune_llama31_written_in_jax_for_free_on_google/)
    + "We’re building AI infra to fine-tune and serve LLMs on non-NVIDIA GPUs (TPUs, Trainium, AMD GPUs)"
  - Reddit post by creators : [AI infra for non-NVIDIA GPUs (and our JAX journey)](https://www.reddit.com/r/LocalLLaMA/comments/1fj9hea/)


### TPU training (Node-style TPUs = old, including Colab)

* [Gemma Inference on TPUs](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/gemma_inference_on_tpu.ipynb)
  - Colab (free version) works with TPU v2-8
  - `from gemma import transformer as transformer_lib`
  - Short and sweet!

* [Unlocking Gemma's Power: Data-Parallel Inference on TPUs with JAX](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/gemma_data_parallel_inference_in_jax_tpu.ipynb)
  - Colab (free version) works with TPU v2-8
  - Pure JAX.flax : [gemma model loaded as `flax`](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.FlaxGemmaForCausalLM):
    + ```
      model, params = FlaxGemmaForCausalLM.from_pretrained(
        model_id, revision="flax", _do_init=False, 
        dtype=jnp.bfloat16, token=access_token)
      ```
* [Fine-tuning Gemma with Torch XLA and Hugging Face TRL](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/Finetune_with_Torch_XLA.ipynb)
  - 2024-12-XX 
  - Colab (free version) works with TPU v2-8
  - Loads `torch-cpu` and `torch_xla[tpu]`
  - Actual training uses `SFTTrainer` (not RL, but from `trl`)

* [JAX Gemma on Colab TPU](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/jax_gemma.ipynb)
  + Colab (free tier) TPU v2-8 (3 generations old)
    - "Gemma itself was trained using JAX on TPU v5e cores"
  + Gemma 2B model in JAX: TPU v2 cores run batched generation with a throughput of 475 tokens per second
    - The Transformers generate method provides functionality for auto-regressive generation with batching, sampling, beam-search, etc. 
    - To reap the benefits of JAX, we'll compile the generate method end-to-end, such that the operations are fused into XLA-optimised kernels and executed efficiently on our hardware accelerator.


### TPU training (VM-style TPUs = modern)

* Google Docs: 
  + [Training with TPU accelerators](https://cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm)
  + [Cloud TPU v5p training](https://cloud.google.com/tpu/docs/v5p-training)
  + [TPU pricing](https://cloud.google.com/tpu/pricing?hl=en)
* graphcast : [Provisioning a Cloud VM TPU](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md)
  + Describes (in detail) how to run `gencast_demo_cloud_vm.ipynb` through Colaboratory using Google Cloud compute
  + == Weather models

