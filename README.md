# Getting to Aha
## With TPU(s) using JAX `nnx`

* Reasoning-from-Zero : JAX on TPU

> Following the release of DeepSeek's R1 model, there was a nice follow-up from a group at Berkeley with a 'Countdown task reasoner' that can be trained from scratch for "$30 of H100s" (https://github.com/Jiayi-Pan/TinyZero).  
> This project is to replicate that same task, but using JAX/TPU infrastructure (hopefully with Gemma as the base model).  
> This will make it far, far more likely that TPUs could become an experimentation platform for the curious : The current barriers to entry are very high.


## R1 Notes

* https://x.com/Guodaya/status/1886635010251518330 
  + DeepSeek Researcher
  + The 660B R1-Zero and R1 began running after the release of V3, 
    with training taking approximately 2-3 weeks. 
    The R1 model we referred to prior to this time (e.g., in the V3 tech report) 
    was the R1-Lite or the R1-Lite-Zero.


## Resources

### Jax (generic)

* https://jax-ml.github.io/scaling-book/
  + DeepMind : Highly recommended

* [Flow Matching in 50 lines of JAX](https://x.com/cgarciae88/status/1867340873136038293)
  + Cristian Garcia at DeepMind OS : @cgarciae88

* Google Docs: 
  + [Fine-tuning `gemma-2b-it` using JAX and Flax](https://ai.google.dev/gemma/docs/jax_finetune)
    - Colab-Pro A100 GPU
      + Could also use Kaggle's (free) TPU v3-8  
      + Colab (free) TPU v2 is insufficient
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


### TPU training (VM-style TPUs)

* Google Docs: 
  + [Training with TPU accelerators](https://cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm)
  + [Cloud TPU v5p training](https://cloud.google.com/tpu/docs/v5p-training)
  + [TPU pricing](https://cloud.google.com/tpu/pricing?hl=en)
* graphcast : [Provisioning a Cloud VM TPU](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md)
  + Describes (in detail) how to run `gencast_demo_cloud_vm.ipynb` through Colaboratory using Google Cloud compute
  + == Weather models



### Post-R1 GRPO demos

* willccbb/grpo_demo.py
  + Will Brown
  + https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
    - from peft import LoraConfig
    - from trl import GRPOConfig, GRPOTrainer
    - beta: (float, optional, defaults to 0.04) — KL coefficient
      + Commenter had success with beta=0.01
      + https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig.beta

* R1-V: Reinforcing Super Generalization Ability in Vision Language Models 
  + Liang Chen = @liangchen5518
    - https://x.com/liangchen5518/status/1886171667522842856
  + https://github.com/Deep-Agent/R1-V  
    - Cost&lt;\$3 : 8 A100 GPUs for 30 minutes
    - 100 training steps

* The Thought Process Behind Kimi k1.5 
  + [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
  + [Informative Author Thread](https://x.com/Kimi_Moonshot/status/1882413059513471044)


### GRPO expositions

* [GRPO with Verifiable (Binary) Rewards Is an Adaptive Weighted Contrastive Loss](https://ymroueh.me/post/post_1/)
  + IBM researcher : Breaks down whitening into the factors
* [Nathan Lambert Book on Blog](https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1)

* [GRPO from DeepSeek-R1 is now available in Hugging Face `trl` library](https://x.com/Hesamation/status/1882001485636178414)

* [Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU](https://huggingface.co/blog/trl-peft)
  + Includes PEFT and `trl` (9-March-2023)

### GRPO libraries

* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
  + High-performance RLHF framework built on Ray, DeepSpeed and HF Transformers
* [veRL](https://github.com/volcengine/verl)
  + Volcano Engine Reinforcement Learning for LLM


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
