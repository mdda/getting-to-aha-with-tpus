# Getting to Aha
## With TPU(s) using JAX `nnx`


## Resources

### Jax (generic)

* https://jax-ml.github.io/scaling-book/
  + DeepMind : Highly recommended

* [Flow Matching in 50 lines of JAX](https://x.com/cgarciae88/status/1867340873136038293)
  + Cristian Garcia at DeepMind OS : @cgarciae88


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


## Potential next ideas

### Agentic RAG

* [Eliciting Critical Reasoning in Retrieval-Augmented Language Models via Contrastive Explanations](https://arxiv.org/abs/2410.22874)



### Task Vectors

* https://x.com/chrisbarber/status/1885047105741611507
  + Shannon Sands (@max_paperclips) from @NousResearch
  + backtracking vector 
    - "caused the chain of thought to backtrack much more often, 
      and when suppressed caused it to be a linear and much shorter CoT"
 

<!--
### Cryptic Crosswords
!-->
