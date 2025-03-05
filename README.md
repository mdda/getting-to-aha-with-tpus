# Getting to Aha
## With TPU(s) <strike>using JAX nnx</strike>

* Reasoning-from-Zero using TPUs for compute
  + Following the release of DeepSeek's R1 model, there was a nice follow-up from a group at Berkeley with a 'Countdown task reasoner' that can be trained from scratch for "$30 of H100s" (https://github.com/Jiayi-Pan/TinyZero)
  + The aim of this project is to replicate that same task, but using a gemma2 model, and TPU infrastructure
  + This will make it far, far more likely that TPUs could become an experimentation platform for the curious : The current barriers to entry are very high


## The Plan

* Use `gemma2-2B-base` on:
  + Kaggle TPU v3-8; and 
  + Colab TPU v2-8 (potentially - it would be very tight)
* Reasoning task : Countdown task from TinyZero
  + RL objective : GRPO
* Goal : Get to "Aha!" using \$free TPU resources
  + with codebase that is:
    * Plain and Readable (i.e. not simply dressing up a call to `trl`)
    * Hackable (i.e. can implement more than the demo case)


### Decision : Which framework?

* [JAX `flax.nnx` examples/gemma](https://github.com/google/flax/tree/main/examples/gemma) (i.e. *new* style)
  + Positives: 
    - Framework being promoted as PyTorch-user-friendly
  + Negatives:
    - Early days (PROVEN)
    - `gemma` example in [`nnx` documentation](https://flax.readthedocs.io/en/latest/guides/gemma.html) does not work
      * [PR submitted to fix glaring error(s)](https://github.com/google/flax/pull/4587)
    - `nnx.jit` of Transformer forward pass proven to take &gt;60Gb RAM during compilation
      * (it would only not crash the VM if the instance had &lt;70Gb available RAM)
      * Therefore impractical for use on Colab/Kaggle == DEAD END
* [Google-DeepMind `gemma` library]() in JAX `flax.linen` (i.e. *old* style)
  + Positives:
    - The library actually works with Gemma2
      * And consumes &lt;1Gb RAM doing `jit` on forward pass / sampling
    - Library has LoRA and sharding
  + Negatives:
    - Flax/linen is (according to the `nnx` docs) backward-looking
    - Heavy dependency on `kauldron` for training (and
      [LoRA](https://github.com/google-deepmind/gemma/blob/main/examples/lora.py#L53),
      [sharding](https://github.com/google-deepmind/gemma/blob/main/examples/sharding.py#L44), etc)
      * Undermines the goal of using plain, readable code 
    - GDM `gemma` library transformer [Sampler is greedy-only](https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py#L145) 
      * Monkey-patching this functionality (which is deep inside the `Sampler` class) would smell bad
      * So adding library features would have to be done before beginning
* [`pytorch-gemma`](https://github.com/google/gemma_pytorch/) library for PyTorch/XLA
  + Positives:
    - Library appears ready for CPU, GPU and TPU
    - Includes distribution code (with Column-wise and Row-wise Linear implementations)
    - Includes 8-bit quantised code
  + Negatives:
    - Does not appear to include LoRA
      * Though may be compatible with PEFT (needs testing)
      * How does auto-LoRA interact with sharding?  Eeek
    - While PyTorch XLA is clearly ['real'](https://github.com/google/gemma_pytorch/blob/main/scripts/run_xla.py#L33) ...
      * Need to test whether XLA code can get 'compiled' in a similar way to JAX `jit`  
* [Keras gemma implementation](https://keras.io/keras_hub/api/models/gemma/gemma_causal_lm/) using JAX backend
  + Positives:
    - Ecosystem appears ready for CPU, GPU and [TPU](https://www.kaggle.com/code/matthewdwatson/gemma-2-tpu-fine-tuning)
    - Includes LoRA, more sophisicated sampling and distribution over TPUs
    - Actually *proven to work* on TPUs via Colab (in this Repo)
  + Negatives:
    - IMHO, Keras is perceived as being somewhat *lame* vs other frameworks
    - Still need to test whether fancy sampling, fancy distribution strategy, and custom training step (GRPO)
      can be implemented *at the same time*

So far: 
* `nnx` has suceeded in:
  + causing me to labouriously debug and fix the example library 
  + wasting many GPU hours frustratedly trying to `nnx.jit` things without crashing the VM
* `gemma` (GDM library) 
  + only has a greedy Sampler - which would need fixing
  + relies very heavily on `kauldron` to do fancy things
* PyTorch/XLA `pytorch_gemma` looks interesting, though would need:
  + LoRA to be added (ideally using PEFT)
  + actual benchmarking on TPUs vs JAX (time-consuming)
* Keras.JAX seems likely to be a good basis,
  + and has *started to show signs of life*
  + though it remains to be seen whether it works as advertised as the model/RL gets more complex

--- 

## Installation / Running the code

```bash
sudo snap install astral-uv --classic
uv venv env_flax.nnx
. ./env_flax.nnx/bin/activate
uv pip install jupyterlab ipywidgets jupytext OmegaConf
```

* Run jupyterlab notebook enviroment:
```bash
jupytext --set-formats cache-notebooks//ipynb,py:light *.py
#...
jupyter-lab --port 8282 --no-browser
```

* Test the countdown puzzle generation:
```bash
pushd ./aha_dataset/countdown/
python generator.py 
popd
```

---

## RL-related Resources

### Post-R1 GRPO demos

* [Experience the Ahah moment yourself for &lt;\$30](https://github.com/Jiayi-Pan/TinyZero)
  + Berkeley : Jiayi Pan=@jiayi_pirate, @JunjieZhang12, @xingyaow_, @lifan__yuan
  + [Author Twitter thread](https://x.com/jiayi_pirate/status/1882839370505621655)
  + TinyZero is a reproduction of DeepSeek R1 Zero in countdown and multiplication tasks
    = We built upon `veRL`
  + CountDown: a game where players combine numbers with basic arithmetic to reach a target number
    + [Scoring Function](https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py#L59), 
    [Dataset with correct answers](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) loaded [here](https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py)
    + This is a bit strange, since the rows of the dataset does not include `{+,-,*,/}` answers
      - ... presumably there is a way to get the `target` from the `nums` ...
      - Maybe: [generated by exhuastive search](https://github.com/kanishkg/stream-of-search/blob/main/src/countdown_generate.py)
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
  + Will Brown = @willccbb
  + Code gist, with comments from implementers / testers
    - Llama 1B, GSM8k
    - `from peft import LoraConfig`
    - `from trl import GRPOConfig, GRPOTrainer`
    - beta: (float, optional, defaults to 0.04) — KL coefficient
      + Commenter had success with beta=0.01
      + https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig.beta
  + (updated code, running smooth now on Qwen-1.5B w/ longer max_completion_length + higher num_generations)
  + "TRL GRPO has vLLM now btw + it's soooo much faster wow"
  + [Next version (?) uses TRL_GRPOTrainer](https://x.com/willccbb/status/1886243810323148890)
  + [Colab version with Qwen 0.5B](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)
    - Runs `vLLM` on Colab GPU too
    - Decent looking code, but Aha is not directly visible...
  + This used by [@anton](https://x.com/abacaj/status/1884361852349825444)
    - [Qwen2.5-0.5B (base model) directly goes into step by step breakdown with zero prompting](https://x.com/abacaj/status/1888826994248323354) (and Llama doesn't produce step-wise thinking of its own accord)
    - [when reward starts going up at step &gt;100 it's either hacking it or discovered something](https://x.com/abacaj/status/1889739663855743472)
    - See below...

* @anton experiments
  + ["Perfect Reward Function"](https://x.com/abacaj/status/1884787697408999619)
  + ["Finished a run (R1 style) GRPO on Qwen-2.5-0.5B (base model) yield +10 accuracy points on GSM8K. Literally just works"](https://x.com/abacaj/status/1885517088304857197)
    - Two tricks I found work well : 
      * use a good system prompt, and 
      * try lower beta (KL coefficient). 
    - 3 rewards: int reward, final_answer tags, and correctness reward
    - has commented on original `willccbb/grpo_demo.py` gist
      + Has own gist of [GRPOTrainer to run gsm8k eval during training](https://gist.github.com/abacaj/9a567910c1a8663f7aa04520075e0ba8)
  + ["Got a better result on qwen2.5-0.5b (base) &rarr; 56% gsm8k"](https://x.com/abacaj/status/1886308242814320834)


* [Full GRPO fine-tuning Qwen2.5 0.5B on a single T4](https://gist.github.com/qunash/820c86d1d267ec8051d9f68b4f4bb656)
  + @qunash on GitHub = https://huggingface.co/anzorq 
  + Fork of the TRL repo by [GitHub @andyl98](https://github.com/andyl98) - with more optimisations
  + `Qwen2.5-0.5B-Instruct` gsm8k eval result from 22.4% to 48.6% 
    - in just ~150 steps (~30 minutes) on a single T4 GPU


* [Train your own R1 reasoning model with Unsloth](https://unsloth.ai/blog/r1-reasoning)
  + [Daniel Han (unsloth) thread](https://x.com/danielhanchen/status/1887564724071768529)
    - We removed double memory usage during vLLM serving and finetuning
    - 70% less VRAM finetuning and 20x faster inference all in one package! 
    - LoRA / QLoRA also originally *did not work* for people when doing GRPO in the starter script
  + [unsloth thread](https://x.com/UnslothAI/status/1887562753126408210)
  + [GRPO with unsloth on free colab](https://colab.research.google.com/drive/1P7frB3fjMv6vjSINqiydAf6gnMab2TiL?usp=sharing)
    - "it's painfully slow; but works :p"
    - Exposes code from TRL training loop a little...
    - `model="Qwen/Qwen2-0.5B-Instruct", reward_funcs="weqweasdas/RM-Gemma-2B",` ... reward model?
  + [Commentary](https://x.com/Hesamation/status/1888285721863004411)
    - GRPO is now optimized to use 80% less VRAM
    - GRPO now with LoRA and QLoRA
    - Qwen2.5(1.5B) can be trained with just 7GB!
    - Llama3.1(8B) training with 15GB


* SimpleRL : [7B Model and 8K MATH Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient](https://hkust-nlp.notion.site/simplerl-reason)
  + Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, Junxian He=@junxian_he
 : hkust
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

* [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
  + Add 'Wait!' when model wants to do '&lt;/think&gt;' to extend thought process
  + SFT on thought traces from ...?
  + [s1: The \$6 R1 Competitor?](https://timkellogg.me/blog/2025/02/03/s1)
    - [Entropix Tie In](https://timkellogg.me/blog/2024/10/10/entropix) - in entropix, extra 'encouragement' tokens were added in... So: similar idea
  + [repo on GitHub](https://github.com/simplescaling/s1)
  + [Project Page](https://simplescaling.github.io/)
  + Frugality:
    - Sifted their dataset of 56K examples down to just the best 1K, 
    - the core 1K is all that's needed to achieve o1-preview performance on a 32B model.
    - Adding data didn't raise performance at all.
  + [s1.1 : trained on same 1K questions](https://x.com/Muennighoff/status/1889310803746246694)
    - DeepSeek answers, rather than Gemini generations
    - As it is just 1K examples, training is extremely cheap and took just 26 minutes
    - To control test-time compute, we develop “budget forcing”:
      * We either force the model to end its thinking or 
      * extend it by appending Wait when the model tries to stop
      * This simple method improves our model
  + GDE Blogpost : [s1 and s1.1](https://gonzoml.substack.com/p/s1-simple-test-time-scaling)


#### Contrarian Ideas

* [There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study](https://oatllm.notion.site/oat-zero)
  + SEA AI Labs in SG (!)
  + [OAT-Zero code on GitHub](https://github.com/sail-sg/oat-zero)
  + Key points:
    - "We found Aha moment (such as self-reflection patterns) appears at epoch 0, namely base models"
    - "Superficial Self-Reflection (SSR) from base models' responses" - leading to wrong answer
    - "increasing response length phenomenon not emergence .. but RL optimizing rule-based reward"
  + [OAT RL library](https://github.com/sail-sg/oat) - A research-friendly framework for LLM online alignment



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


### GRPO Hints
 
* [DeepSeek R1 training is straight-forward, UNTIL you understand the complexities in writing GRPO Verifiers](https://x.com/bookwormengr/status/1888530568645861865)
  + Somewhat ranty

* Trellis video series:
  + 1: [Reinforcement Learning for LLMs in 2025](https://www.youtube.com/watch?v=C4HxJQ2QzWo)
    - Set-up of training, with curation of SFT data (mostly)
  + 2: [How does GRPO work?](https://www.youtube.com/watch?v=iHlarYGLMbY)
    - 32mins : TODO:WATCH!

* [GRPO implementation update](https://github.com/allenai/open-instruct/issues/534#issuecomment-2634656168)
  + Fixing up the implementation in AllenAI RL library
  + [Other comments](https://x.com/vwxyzjn/status/1885329398821187633):
    - When directly minimizing the KL loss, kl3 just appears much more numerically stable. 
    - And the &gt;0 guarantee here is also really nice (kl1 could go negative).
  + [John Schulman's Homepage : Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
  + BUT ... [LMs with GRPO etc with KL penalty = 0 works](https://x.com/natolambert/status/1890071898869907646)
    - "These are from experiments and this is not official training advice."


### GRPO libraries

* [GRPO from DeepSeek-R1 is now available in Hugging Face `trl` library](https://x.com/Hesamation/status/1882001485636178414)
  + [`GRPOTrainer` Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
  + KL divergence is *estimated* using the approximator introduced by Schulman et al. (2020)
    - The approximator is defined as follows:  `p_ratio - log(p_ratio) - 1`
  + Has a `use_vllm=True` parameter to do generations using `vllm`
  + ["just a reminder : trl grpo is not same as same as described in deepseek paper"](https://x.com/shxf0072/status/1886390053104242983)
    - No clipping objective (though does have KL term) (may not be important at all)
      + Maybe [KL term not needed with verifiable rewards](https://x.com/shxf0072/status/1892687698261139566)
    - Also "Joey (e/λ)" has [comments about gradient / loss and removing constants](https://x.com/shxf0072/status/1892668791303373042)...  
      + Claim : "loss =  advantage*log_softmax(logits) works, same gradients"
      + (Makes sense at first glance, but not clear whether there's something else going on)
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
  + High-performance RLHF framework built on Ray, DeepSpeed and HF Transformers
* [veRL](https://github.com/volcengine/verl)
  + Volcano Engine Reinforcement Learning for LLM


### R1 Notes

* https://x.com/Guodaya/status/1886635010251518330 (now deleted)
  + =Researcher at DeepSeek 
  + The 660B R1-Zero and R1 began running after the release of V3, with training taking approximately 2-3 weeks
  + The R1 model prior to this time (e.g., in the V3 tech report) was the R1-Lite or the R1-Lite-Zero


### Miscellaneous

* [GRPO VRAM Requirements For the GPU Poor](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor)
  + Points out RAM requirements (with potential torch ideas)
  + GRPO explanation not very useful

* [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
  + 2023-10-24

* [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
  + 2022-03-25


---

## Potential next ideas

### RL on Deepseek 'hard distilled' models

* [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
  + Berkeley Sky Computing Lab (not the same authors as original \$30 one, AFAICT)
  + "1.5B model beats o1-preview on math by RL"
  + Cost:
    - Overall, our training run consists of ~1,750 steps. 
    - The initial 8K context phase was trained on 8 A100 GPUs, 
    - while the 16K and 24K phases scaled up training to 32 A100 GPUs. 
    - In total, the training took ~3,800 A100 hours = roughly 5 days on 32 A100s
    - \$4500 in compute cost
  + Reddit discussion [DeepScaleR-1.5B-Preview](https://www.reddit.com/r/LocalLLaMA/comments/1imm4wc/deepscaler15bpreview_further_training/)
  + [Model on HF](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
  + [Project on GitHub](https://github.com/agentica-project/deepscaler) 
    - uses their own [veRL](https://github.com/agentica-project/verl)
  + ["DeepScaleR is by far the most sophisticated and impressive thing built on R1 this far"](https://x.com/teortaxesTex/status/1889914611555865007)
    - Maximizing intelligence per FLOP is a natural step after test time unlock



### Agentic RAG

* [Agentic RAG systems and taxonomy](https://x.com/tom_doerr/status/1889905154465448265)
  + [Actual repo](https://github.com/asinghcsu/AgenticRAG-Survey)

* [Eliciting Critical Reasoning in Retrieval-Augmented Language Models via Contrastive Explanations](https://arxiv.org/abs/2410.22874)
* [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
* [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342)
  + Microsoft
  + More than 10 points improvement in EM score compared to strong baseline
  + Establishes a new SotA performance across a diverse range of knowledge-intensive tasks


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


### Agent RL

* [Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning](https://arxiv.org/abs/2302.02662)
  + T5 (in 2023-02)
* [RAGEN: A General-Purpose Reasoning Agent Training Framework](https://github.com/ZihanWang314/ragen)
  + Code on GitHub
  + [Author Thread](https://x.com/wzihanw/status/1884092805598826609)
  + We run RAGEN on the Gym-Sokoban task: 
    - Qwen-2.5-{0.5B, 3B}-{Instruct, None}
    - DeepSeek-R1-Distill-Qwen-1.5B
* [Scaled Cognition: "first ever models trained specifically for agentic applications"](https://x.com/ScaledCognition/status/1889721166421479751)
  - "APT-1, is now #1 on agentic benchmarks" ...


### Task Vectors

* https://x.com/chrisbarber/status/1885047105741611507
  + Shannon Sands (@max_paperclips) from @NousResearch
  + backtracking vector 
    - "caused the chain of thought to backtrack much more often, 
      and when suppressed caused it to be a linear and much shorter CoT"

<!--
### Cryptic Crosswords
!-->

---

## TPU Resources

See [this page](TPU.md) for:
* JAX Resources
  + JAX (generic)
  + Gemma Models
  + Keras (JAX backend)
  + LoRA for JAX
* TPU training
  + TPU training (Node-style TPUs = old, including Colab)
  + TPU training (VM-style TPUs = modern)


