
* nathan lile @NathanThinks
  + [Comment](https://x.com/NathanThinks/status/1897037977756602786)
    - Qwen+RL = dramatic, Aha! ; Llama+RL = quick plateau ; Same size. Same RL. Why?
    - Qwen naturally exhibits cognitive behaviors that Llama doesn't
    - Prime Llama with 4 synthetic reasoning patterns & it matched Qwen's self-improvement performance!
  + [Comment](https://x.com/NathanThinks/status/1897037980604489797)
    - Surprisingly, models primed with incorrect solutions but the correct reasoning processes perform
      just as well as those trained on correct solutions
    - [Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs](https://arxiv.org/abs/2503.01307)

 * [Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought](https://arxiv.org/abs/2501.04682)
   + Meta-CoT

* You Jiacheng
  + [... why do they use mean instead of log(sum(exp(...))), the extra cost seems negligible](https://x.com/YouJiacheng/status/1881403115909566905)


* Jason Weston
  + [Diverse Preference Optimization (DivPO)](https://arxiv.org/abs/2501.18101)
    - Nice graph with SFT and RL learning curves
    - [Thread](https://x.com/jaseweston/status/1885399530419450257)

* kalomaze
  + [Comment](https://x.com/kalomaze/status/1890688569863254450)
    - WTF, is TRL GRPO KL divergence not per token mean and instead per sequence mean (????)
    - taking the average per sequence is very bad if your completions range from 4 tokens to 400
  + [Comment](https://x.com/kalomaze/status/1890633787458908418)
    - @willccbb repo was my first exposure to doing full finetuning GRPO, 
    - and it took me under ~2 hours to get a basic grip on modifying reward funcs 
    - w/ trivial dependency or abstraction wrangling.
    - already a solid repo so far
    - NB: 14B Full FT works just fine on 8xH100s. Everyone else so far that I've seen is doing like, ~3B
  + [Comment](https://x.com/kalomaze/status/1890953337933074489)
    - min_p sampling of 0.01 -starts worse- (for formatting reward especially), but ascends faster
    - lightweight modification to how vllm sampling params are passed in @willccbb's excellent WIP repo (verifiers)
  + [Comment](https://x.com/kalomaze/status/1891655122414882911)
    - turns out you can make the KL term several orders of magnitude more stable 
    - by sampling a couple hundred tokens stochastically and only caring about that subset...
    - rather than a majority of comparisons being within the "floating point precision error" range
      * DeepSeek's GRPO uses forward KL divergence instead of reverse KL divergence (which PPO does). 
      * tried implementing the reverse kl term, and... hm.)
  + [Comment](https://x.com/kalomaze/status/1891799888028946883)
    - an overwhelming amount of extreme outlier KL divergences in my GRPO training comes from the formatting tokens
    - because it is (correctly) becoming deterministic for the XML wrapping / turn ending
  + [Blog Post](https://kalomaze.bearblog.dev/grpo-judge-experiments-findings-and-empirical-observations/)
    - [Announcement](https://x.com/kalomaze/status/1896699353710092533)
  + [Comment]()


* Joey (e/λ) @shxf0072
  + [Comment](https://x.com/shxf0072/status/1894451012309848339)
    - just a reminder : TRL GRPO is not same as same as described in deepseek paper
      * It doesn't have clipping objective, which is key innovation in PPO,
      * GRPO has clipping + KL; TRL just have KL
    - ... gets discussed/fixes proposed
  + [Comment](https://x.com/shxf0072/status/1892668791303373042)
    - even if exp(0) = 1 gradient still flows trough this part of graph
    - ... derivation showing derivatives can be greatly simplified
    

* Will Brown @willccbb
  + [Comment]()

* Daniel Han @danielhanchen (unsloth)
  + [Comment](https://x.com/danielhanchen/status/1892643424538595611)
    - We further slashed memory usage of GRPO (the algorithm behind R1) in @UnslothAI to 90% savings!

* Teortaxes @teortaxesTex
  + [Comment](https://x.com/teortaxesTex/status/1892505232099193208)
    - Both KL Loss and KL Penalty mechanisms not only slow down the training process 
    - but also consume computational resources that could be better utilized for reward optimization
    - [Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://arxiv.org/abs/2503.24290)
      * [Code on GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) (MIT)
      * [Author Thread](https://x.com/CyouSakura/status/1892428094075502960)


* Tassel Pierre @tassel_pierre
  + [Comment](https://x.com/tassel_pierre/status/1891502229095354765)
    - You can even remove the beta and just go with low grad clipping, 
    - it has for advantage that you do not need any ref model then (faster and less memory consumption):
    - [GRPO - Do not load reference model when beta == 0](https://github.com/huggingface/trl/pull/2806)


* Alex Dimakis
  + Discovered a very interesting thing about DeepSeek-R1 and all reasoning models: 
    - The wrong answers are much longer while the correct answers are much shorter.

* Yunzhen Feng
  + [PILAF: Optimal Human Preference Sampling for Reward Modeling](https://arxiv.org/abs/2502.04270)
    - On-policy sampling required for best reward models? Think again! 
    - [Thread](https://x.com/feeelix_feng/status/1888279449621107146)


* [People nailing “Aha moment” :](https://x.com/Xianbao_QIAN/status/1889196451416412472)
  + Running Open R1 on a surprisingly small 0.5B model. 
    - And it only need 4x4090 to work with, proving you don’t always need a massive compute setup to spark innovation.


* Maziyar PANAHI
  + Running DeepSeek’s GRPO on SMOLLM2-1.7B model from @huggingface! Not perfect yet, but we’re making progress.
    - [Thread](https://x.com/MaziyarPanahi/status/1890384650125652213)
    - This was the gsm8k with a small mix from some math reasoning. 
    - But tonight I will run 10k-20k from OpenR1-Math-220k to see what happens:


* Costa Huang
  + allenai/Llama-3.1-Tulu-3-8B (trained with PPO) -> allenai/Llama-3.1-Tulu-3.1-8B (trained with GRPO) 
  + Our latest GRPO-trained Tulu 3.1 model, which is considerably better in MATH and GSM8K!
    - [Thread](https://x.com/vwxyzjn/status/1889728091401973968)

* [REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models](https://arxiv.org/abs/2501.03262)
  + [Blog Post](https://hijkzzz.notion.site/reinforce-plus-plus)
  + Related:
    - [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
    - [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2502.14768)

* [Reinforcement Learning is all You Need](https://arxiv.org/abs/2503.09512)
  + 3 billion parameter language model + The Countdown Game dataset 
  + "Aha moments" and response length versus reasoning are analyzed

* [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
  + SEA-AI (Singapore)
  + [Code Repo](https://github.com/sail-sg/understand-r1-zero) (MIT)
  + [Author Thread](https://x.com/zzlccc/status/1903162768083259703)
    - DeepSeek-V3-Base already exhibits "Aha moment" before RL-tuning??
    - The ever-increasing output length in RL-tuning might be due to a BIAS in GRPO??
    - Getting GRPO Done Right, we achieve a 7B AIME sota!
  + [Another Author Thread](https://x.com/Cameron_Chann/status/1903167286321795532)
  + [News Release](https://x.com/Marktechpost/status/1903669828776566948)
  + [Promoting Thread](https://x.com/WenhuChen/status/1903464313391624668)
    - Previously, people found that Qwen base models are particularly good at R1 training to show strong exploration skills. 
      * This paper shows that there is no magic about Qwen base models. 
      * It's likely pre-trained with concatenated Q+A data. 
      * Therefore, the base models will automatically answer questions instead of completing it. 
      * Therefore, pre-training LLama-3.2 a bit on similar concatenated Q+A data can also trigger it to explore and achieve better performance.
    - Previously, there is a common belief that "Aha Moment" is a result from RL training.
      * This paper shows that some base models already exhibit amount of self-reflection. 
      * RL is simply enhancing this behavior.
    - Previously, the increased output length is believed to be the key for performance improvement.
      * This paper argues that it's not the case. 
      * The responses with self-reflection get lower accuracy than the ones without self-reflection.
    - Previously, people are obsessed with length increase with GRPO algorithm.
      * This paper argues that this phenomenon is simply due to the length bias in GRPO. 
      * Basically, by dividing the advantage with the total length, the wrong responses are penalized less than shorter responses in GRPO. 
      * By removing the length normalization term, the length won't increase dramatically while the performance even increases slightly.
  + [Kawin Ethayarajh comment on division 'bug'](https://x.com/ethayarajh/status/1903859350021812698)
    - If you use TRL or verl, you are fine, no changes needed.  This is an OpenRLHF-specific bug. 
    - If you use TRL/verl, although the masked_mean function supports taking the mean within a sequence, 
      this option isn't invoked when calculating the PPO loss.
    - Moreover, not normalizing by the number of active tokens in the batch, as in the proposed solution (green), 
      will likely lead to less stable training and preclude the comparison of batch-wise losses.


* [SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild](https://arxiv.org/abs/2503.18892)
  + [Junxian He Thread](https://x.com/junxian_he/status/1904527884934697050)
  + Secret Sauces:
    - Remove the format reward. 
      * Format reward constrains model's exploration, where training on llama3-8B would fail.
    - Choosing properly difficult data for your model. 
      * For example, training Mistral-7B with Math level3-5 data fails.


* [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)
  + [Project Page](https://limit-of-rlvr.github.io/)


* [Language Models can Self-Improve at State-Value Estimation for Better Search](https://arxiv.org/abs/2503.02878)
  + Self-taught lookahead, a self-supervised method that leverages state-transition dynamics 
    - to train a value model capable of effectively guiding language model-controlled search
  + Improves performance by 20% while reducing costs 37x compared to previous LLM-based tree search, 
    - without relying on ground truth rewards
  + "First demonstration that an LLM-based value function can self-improve without labels or rewards"
  + [Code Coming Soon](https://github.com/ethanm88/self-taught-lookahead)


* [ϕ-Decoding: Adaptive Foresight Sampling for Balanced Inference-Time Exploration and Exploitation](https://arxiv.org/abs/2503.13288)
  + [Posted on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1jfrwqw/new_sampling_method_that_boosts_reasoning/)
  + [Repo on GitHub](https://github.com/xufangzhi/phi-Decoding)


* Hrishbh Dalal @HrishbhDalal (AI Researcher and Freelancer in Germany)
  + [Comment](https://x.com/HrishbhDalal/status/1896955224541290926)
    - 3B parameter model beat GPT-4, Claude, and other giants at financial sentiment analysis?
    - 84% accuracy on a complex multilingual task with minimal compute
    - thanks to @willccbb @kalomaze and @UnslothAI for their (teachings, notes, communication) and (code) respectively


* Nathan Lambert @natolambert
  + [Comment](https://x.com/natolambert/status/1900639281791615387)
    - Does anyone have an intuition or ablation on applying the KL penalty in the loss directly 
      rather than when the reward is computed? How is this changing learning?
    - normal    : rewards = rewards - self.beta * per_token_kl 
    - GRPO impl : per_token_loss = pg_loss_max + self.beta * per_token_kl
    - ... fair amount of useful discussion...
  + [Learning about GRPO](https://x.com/natolambert/status/1903996949377995013)
    - [YouTube video](https://www.youtube.com/watch?v=amrJDwMUFNs)
    - [Daniel Han also wonders about divisor 'bug'](https://x.com/danielhanchen/status/1904088139741970760)
  + [Pretty comprehensive research survey of recent papers I liked](https://x.com/natolambert/status/1906739705733099791)
    - Kimi 1.5, OpenReasonerZero, DAPO and Dr. GRPO
    - [Recent reasoning research: GRPO tweaks, base model RL, and data curation](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo)
  + [Comment]()


* Teknium (e/λ)  @Teknium1
  + [Comment](https://x.com/Teknium1/status/1901135321745813721)
    - Tool Calling RL achieved internally

* [TRL update makes GRPO 60x faster](https://x.com/jeffboudier/status/1904294208275624123)
  + This all comes in the v0.16 update of TRL that shipped today...
    - multi-step optimization for 6x speedup
    - global normalization from Dr GRPO
    - domain-specific rewards
    - beta=0.0 memory savings
    - padding-free batching for SFT
  + [Release Notes](https://github.com/huggingface/trl/releases/tag/v0.16.0)


* Amirhossein Kazemnejad @a_kazemnejad
  + [Announcement](https://x.com/a_kazemnejad/status/1907849729863471204)
    - nanoAhaMoment: Karpathy-style, single file RL for LLM library (<700 lines)
      * super hackable
      * no TRL / Verl, no abstraction💆‍♂️
      * Single GPU, full param tuning, 3B LLM
      * Efficient (R1-zero countdown < 10h)
        - ... recreated DeepSeek R1-Zero style-training on CountDown in ~10h with *one A100*
    - [YouTube playlist](https://www.youtube.com/playlist?list=PL_vLws1T4Nx1LSx7bXag7mc2IZIgcY6We)
    - [Code Repo](https://github.com/McGill-NLP/nano-aha-moment)

* Qingfeng Lan 
  + [From REINFORCE to Dr. GRPO](https://lancelqf.github.io/note/llm_post_training/)
    - In this blog, I explore their relationships and provide a unified perspective through the Policy Gradient Theorem
    - = the backbone of policy gradient methods.

