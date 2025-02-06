# Getting to Aha
## With TPU(s) using JAX `nnx`

* Reasoning-from-Zero : JAX on TPU
  + Following the release of DeepSeek's R1 model, there was a nice follow-up from a group at Berkeley with a 'Countdown task reasoner' that can be trained from scratch for "$30 of H100s" (https://github.com/Jiayi-Pan/TinyZero).  
  + This project is to replicate that same task, but using JAX/TPU infrastructure (hopefully with Gemma as the base model).  
  + This will make it far, far more likely that TPUs could become an experimentation platform for the curious : The current barriers to entry are very high.


## R1 Notes

* https://x.com/Guodaya/status/1886635010251518330 
  + DeepSeek Researcher
  + The 660B R1-Zero and R1 began running after the release of V3, 
    with training taking approximately 2-3 weeks. 
    The R1 model we referred to prior to this time (e.g., in the V3 tech report) 
    was the R1-Lite or the R1-Lite-Zero.


## JAX Resources

### Jax (generic)

* [Flow Matching in 50 lines of JAX](https://x.com/cgarciae88/status/1867340873136038293)
  + Cristian Garcia at DeepMind OS : @cgarciae88

* Google Docs: 
  + [Fine-tuning `gemma-2b-it` using JAX and Flax](https://ai.google.dev/gemma/docs/jax_finetune)
    - Colab-Pro A100 GPU
      + Could also use Kaggle's (free) TPU v3-8  
      + Colab (free) TPU v2 is insufficient
        * Even though `optimizer = optax.sgd(training_cfg.learning_rate)`
    - The gemma library was written with:
      + JAX, Flax, 
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
    - "Update (4/8/2024): JORA now supports Google's Gemma models"


### Gemma Models

* [Using pretrained Gemma for inference with Flax NNX](https://flax.readthedocs.io/en/latest/guides/gemma.html)
  + Gemma1 : 2.6B + 7B
  + Gemma2 : 2B + 9B + 27B  (2, 8, 13 Trillion tokens training, respectively)
  + CodeGemma : 2B + 7B
  + [Gemma formatting and system instructions](https://ai.google.dev/gemma/docs/formatting)
  + Flax NNX `gemma.transformer.TransformerConfig.from_params` function

* [Getting Started with RecurrentGemma and FLAX](https://www.kaggle.com/code/philculliton/getting-started-with-recurrentgemma-and-flax/notebook)
  + RecurrentGemma : 2B
    - Griffin... : [Fine-tuning the 2B Griffin model with Flax](https://www.kaggle.com/code/philculliton/fine-tuning-the-2b-griffin-model-with-flax)
  + [Fine-tuning RecurrentGemma using JAX and Flax](https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_finetune)
    - `model = recurrentgemma.Griffin(config)`
  + [Inference with RecurrentGemma using JAX and Flax](https://ai.google.dev/gemma/docs/recurrentgemma/recurrentgemma_jax_inference)
  + [flax/examples/gemma/](https://github.com/google/flax/tree/main/examples/gemma) code seems to include `nnx` and `nn`
  

### Keras (Jax backend)

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
    - "Built on modern Flax NNX, it provides convenient and effective solutions for training and serving Flax/Jax models on TPU/GPU at scale"
  + Has `load_in_8bit`, `class DeepseekV3Attention(FlaxAttentionModule):`
  + [LoRA example for SFT `Mistral-7B-Instruct` on Kaggle TPUs](https://www.kaggle.com/code/citifer/easydel-causal-language-model-trainer-example)
  + [Training example in documentation](https://easydel.readthedocs.io/en/latest/supervised-fine-tuning-trainer-example.html)

* [LoRA - NNX LoRA classes](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/lora.html)
* [NNX parameter surgery](https://flax.readthedocs.io/en/latest/why.html#model-surgery)
  + [Model Surgery (less informative)](https://flax.readthedocs.io/en/latest/guides/surgery.html)
  + [Toy Example Code](https://github.com/google/flax/blob/main/examples/nnx_toy_examples/09_parameter_surgery.py)


* [`lorax`](https://github.com/davisyoshida/lorax/blob/master/examples/huggingface_gpt2.py)
  - [May not be the best implementation](https://github.com/jax-ml/jax/discussions/16588)



## TPU training

* [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
  + DeepMind : Highly recommended

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

* [Felafax - tune LLaMa3.1 on Google Cloud TPUs for 30\% lower cost and scale seamlessly!](https://github.com/felafax/felafax?tab=readme-ov-file#-finetune-for-free)
  - Full range of Llama models in Jax; Has LoRA implementation
  - Seems to 'take care of' spinning up everything (i.e. may be over-kill)
  - Reddit post by creators : [Tune LlaMa3.1 (written in JAX) for free on Google Colab TPU](https://www.reddit.com/r/LocalLLaMA/comments/1fj9hea/tune_llama31_written_in_jax_for_free_on_google/)
    + "We’re building AI infra to fine-tune and serve LLMs on non-NVIDIA GPUs (TPUs, Trainium, AMD GPUs)"
  - Reddit post by creators : [AI infra for non-NVIDIA GPUs (and our JAX journey)](https://www.reddit.com/r/LocalLLaMA/comments/1fj9hea/  


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



## RL-related Resources

### Post-R1 GRPO demos

* [Experience the Ahah moment yourself for &lt;\$30](https://github.com/Jiayi-Pan/TinyZero)
  + @jiayi_pirate, @JunjieZhang12, @xingyaow_, @lifan__yuan
  + [Author Twitter thread](https://x.com/jiayi_pirate/status/1882839370505621655)
  + TinyZero is a reproduction of DeepSeek R1 Zero in countdown and multiplication tasks
    = We built upon `veRL`
  + CountDown: a game where players combine numbers with basic arithmetic to reach a target number
  + Tried : Qwen-2.5-Base 0.5B, 1.5B, 3B, 7B
    - &gt;1.5B models start learning to search, to self-verify and to revise solutions
  + Either base or instruct model works 
    - Converge to same performance (instruct learns more quickly)
    - Instruct model's output are more structured and readable
  + PPO, GRPO and PRIME all worked
* [Mini-R1: Reproduce Deepseek R1 "aha moment" - an RL tutorial](https://www.philschmid.de/mini-deepseek-r1)
  + = same as above
  + "This blog is inspired by Jiayi Pan who initially explored the idea and proofed it with a small model."


* [willccbb/grpo_demo.py](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
  + Will Brown
  + Code gist, with comments from implementers / testers
    - Llama 1B
    - from peft import LoraConfig
    - from trl import GRPOConfig, GRPOTrainer
    - beta: (float, optional, defaults to 0.04) — KL coefficient
      + Commenter had success with beta=0.01
      + https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig.beta
  + (updated code, running smooth now on Qwen-1.5B w/ longer max_completion_length + higher num_generations)
  + This used by [@anton](https://x.com/abacaj/status/1884361852349825444)
  + "TRL GRPO has vLLM now btw + it's soooo much faster wow"
  + [Next version (?) uses TRL_GRPOTrainer](https://x.com/willccbb/status/1886243810323148890)
  + [Colab version with Qwen 0.5B](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)
    - Runs `vLLM` on Colab GPU too


* [7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient](https://hkust-nlp.notion.site/simplerl-reason)
  + Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, Junxian He : hkust
    > We reproduce the training of DeepSeek-R1-Zero and DeepSeek-R1 for complex mathematical reasoning, 
    > starting from Qwen-2.5-Math-7B (base model), 
    > and only using 8K (query, final answer) examples from the original MATH dataset. 
  + [Code on GitHub](https://github.com/hkust-nlp/simpleRL-reason)
    - We are working on the paper and will release it very soon
    - Uses OpenRLHF

* ["R1-V: Reinforcing Super Generalization Ability in Vision Language Models"](https://x.com/liangchen5518/status/1886171667522842856)
  + Liang Chen = @liangchen5518
  + https://github.com/Deep-Agent/R1-V  
    - Cost&lt;\$3 : 8 A100 GPUs for 30 minutes
    - 100 training steps

* The Thought Process Behind Kimi k1.5 
  + [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
  + [Informative Author Thread](https://x.com/Kimi_Moonshot/status/1882413059513471044)

* @anton experiments
  + ["Perfect Reward Function"](https://x.com/abacaj/status/1884787697408999619)
  + ["Finished a run (R1 style) GRPO on Qwen-2.5-0.5B (base model) yield +10 accuracy points on GSM8K. Literally just works"](https://x.com/abacaj/status/1885517088304857197)
    - Two tricks I found work well : 
      * use a good system prompt, and 
      * try lower beta (KL coefficient). 
    - 3 rewards: int reward, final_answer tags, and correctness reward
    - has commented on original `willccbb/grpo_demo.py` gist
  + ["Got a better result on qwen2.5-0.5b (base) &rarr; 56% gsm8k"](https://x.com/abacaj/status/1886308242814320834)

* [manged to hack grpo with unsloth on free colab](https://colab.research.google.com/drive/1P7frB3fjMv6vjSINqiydAf6gnMab2TiL?usp=sharing)
  + "it's painfully slow; but works :p"
  + Exposes code from TRL training loop a little...
  + `model="Qwen/Qwen2-0.5B-Instruct", reward_funcs="weqweasdas/RM-Gemma-2B",` ... reward model?

* [S1: The \$6 R1 Competitor?](https://timkellogg.me/blog/2025/02/03/s1)
  + Add 'Wait!' when model wants to do '&lt;/think&gt;' to extend thought process
  + SFT on thought traces from 
  + [Entropix Tie In](https://timkellogg.me/blog/2024/10/10/entropix) - in entropix, extra 'encouragement' tokens were added in... So: similar idea
  + Frugality:
    - Sifted their dataset of 56K examples down to just the best 1K, 
    - the core 1K is all that's needed to achieve o1-preview performance on a 32B model.
    - Adding data didn't raise performance at all.





### GRPO expositions

* [GRPO with Verifiable (Binary) Rewards Is an Adaptive Weighted Contrastive Loss](https://ymroueh.me/post/post_1/)
  + IBM researcher : Breaks down whitening into the factors
* [Nathan Lambert Book on Blog](https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1)

* [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
  + Includes PEFT and `trl` (9-March-2023)

* [GRPO also works very well for Llama 2 7B, with an impressive +15 accuracy point increase in GSM8K](https://x.com/rdolmedo_/status/1886505669622149139)
  + "There's nothing magical about recent model families. If the model can perform some task with sufficient accuracy, then RL with verifiable rewards will likely boost performance"
  + [Run it yourself: `RicardoDominguez/grpo_llama2-7b-chat_gsm8k.sh`](https://gist.github.com/RicardoDominguez/72603d278ed26f0dd55af6ffd414b797)
    - Seems like unrolled code from TRL ... everything is there

   

### GRPO libraries

* [GRPO from DeepSeek-R1 is now available in Hugging Face `trl` library](https://x.com/Hesamation/status/1882001485636178414)
  + [`GRPOTrainer` Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
  + KL divergence is *estimated* using the approximator introduced by Schulman et al. (2020)
    - The approximator is defined as follows:  `p_ratio - log(p_ratio) - 1`
  + Has a `use_vllm=True` parameter to do generations using `vllm`
  + ["just a reminder : trl grpo is not same as same as described in deepseek paper"](https://x.com/shxf0072/status/1886390053104242983)
    - No clipping objective (though does have KL term) (may not be important at all)
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
  + High-performance RLHF framework built on Ray, DeepSpeed and HF Transformers
* [veRL](https://github.com/volcengine/verl)
  + Volcano Engine Reinforcement Learning for LLM

### Miscellaneous

* [GRPO VRAM Requirements For the GPU Poor](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor)
  + Points out RAM requirements (with potential torch ideas)
  + GRPO explanation not very useful



## Potential next ideas

### Agentic RAG

* [Eliciting Critical Reasoning in Retrieval-Augmented Language Models via Contrastive Explanations](https://arxiv.org/abs/2410.22874)
* [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)


#### Datasets

* [BERGEN: A Benchmarking Library for Retrieval-Augmented Generation](https://arxiv.org/abs/2407.01102)
  + [No results yet?](https://paperswithcode.com/paper/bergen-a-benchmarking-library-for-retrieval)
* [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431)
* [LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain](https://arxiv.org/abs/2408.10343)
  + [LegalBench-RAG](https://github.com/ZeroEntropy-AI/legalbenchrag)
* [Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2409.12941)
  + [FRAMES: Factuality, Retrieval, And reasoning MEasurement Set](https://huggingface.co/datasets/google/frames-benchmark)
* [CRAG: Comprehensive RAG Benchmark](https://github.com/facebookresearch/CRAG)
  + [KDD Task - with starter kit](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)
* [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/)
  + [Google : NaturalQuestions](https://ai.google.com/research/NaturalQuestions/dataset)
  + [Natural Questions SoTA](https://paperswithcode.com/sota/question-answering-on-natural-questions)


### Task Vectors

* https://x.com/chrisbarber/status/1885047105741611507
  + Shannon Sands (@max_paperclips) from @NousResearch
  + backtracking vector 
    - "caused the chain of thought to backtrack much more often, 
      and when suppressed caused it to be a linear and much shorter CoT"
 

<!--
### Cryptic Crosswords
!-->
