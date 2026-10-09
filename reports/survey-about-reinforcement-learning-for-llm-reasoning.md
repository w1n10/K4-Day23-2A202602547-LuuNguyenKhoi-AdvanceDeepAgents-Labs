# Survey on Reinforcement Learning for LLM Reasoning

## TL;DR
- Reinforcement learning (RL) shifts LLM reasoning from simple outcome prediction to validating multi-step logical chains [1][2].
- Advanced training frameworks like Group Relative Policy Optimization (GRPO) enable efficient policy learning without requiring costly separate critic models [1][3].
- Test-time compute scaling is a core emerging trend, where models leverage learned strategies or self-reflection to verify and refine outputs [2][4][5].
- Recent approaches prioritize step-level feedback (Process Reward Models) and self-correction to overcome limitations of traditional majority-voting consensus [1][6][7].

## Background
Reinforcement learning for Large Language Models (LLMs) seeks to optimize reasoning capabilities by training models to generate structured intermediate Chain-of-Thought (CoT) steps rather than just final answers [3][8]. Traditionally, LLMs were fine-tuned via Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF) focusing on answer accuracy. Current reasoning-focused RL transitions this paradigm toward "verifiable" reasoning, where intermediate steps are checked against deterministic rules (e.g., compilers, calculators) or learned process reward models (PRMs) [1][3]. This shift allows models to perform complex multi-step reasoning by enabling them to backtrack and self-correct during the thinking process [8][4].

## Foundations of RL for Reasoning
The core objective in RL for reasoning is to optimize the policy for logical soundness rather than just binary outcome accuracy. Approaches range from outcome-based reward models (ORMs), which judge the final answer, to process reward models (PRMs), which offer dense supervision at each reasoning step [1][6]. Verifiable reinforcement learning (RLVR) is a foundational technique that replaces subjective human-derived rewards with objective, rule-based feedback, facilitating training without extensive manual labeling [1][3]. By framing the reasoning task as a Markov Decision Process (MDP), models can learn to navigate complex logical state spaces [2].

## Main RL Methodologies
Policy optimization has seen significant evolution, particularly with methods like GRPO, which computes relative quality advantages among groups of rollouts to avoid the need for separate, compute-heavy critic models [3][2]. Other methodologies address specific training challenges:
- **Optimization Robustness:** Techniques like ReSPO address "gradient starvation" by reshaping sequences, while credit-augmentation adjusts for model sensitivity to prompt features [9][10].
- **Self-Supervision:** Self-rewarding frameworks utilize internal consistency and volatility as signals, enabling models to learn from their own generated reasoning paths without ground-truth labels [11].
- **Distillation:** While on-policy distillation is a common practice, it requires careful implementation to avoid "model collapse" where the model converges prematurely to repetitive, suboptimal paths [12].

## Benchmarks and Performance Evidence
Reasoning capabilities are evaluated across a spectrum of benchmarks, including mathematical (GSM8K, MATH) and agentic construction tasks (BrickBench) [13]. Evidence suggests that RL consistently outperforms standard base-model training by enhancing logical navigation and error detection [14]. Research indicates that log-probability-based rewards often provide more nuanced signals than binary outcomes, leading to better CoT alignment [15]. Furthermore, memory-efficient variants like S-GRPO allow for reasoning-focused RL training even on compute-constrained platforms, demonstrating that robust reasoning is attainable without massive parameter scaling [16][17].

## Trends and Open Problems
The field is currently moving from simple consensus-based voting (majority voting) toward sophisticated, self-reflective reasoning that can identify correct paths even when they represent a logical minority [5]. Key trends include:
- **Test-Time Compute Scaling:** Scaling inference-time compute allows models to explore multiple reasoning paths and iteratively refine them via self-critique [4][7].
- **Exploration Stability:** Dynamic reward shaping is being investigated to prevent "entropy collapse," where models prematurely abandon diverse exploration in favor of local optima [18].
- **Debate on Emergence:** An ongoing question is whether advanced reasoning capabilities are solely induced by RL training or if they are primarily dormant features of pre-trained foundations that RL merely amplifies [4][5].
Open challenges remain in managing the substantial computational costs associated with long-chain reasoning, stabilizing benchmark metrics, and ensuring generalizability across diverse domains [19][4].

## References
[1] Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey. web. https://arxiv.org/html/2510.01925v3 (0000-00-00)
[2] A Tutorial on LLM Reasoning: Relevant Methods behind ChatGPT o1. web. https://www.alphaxiv.org/abs/2502.10867 (2025-02-15)
[3] The State of Reinforcement Learning for LLM Reasoning. web. https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training (2025-04-19)
[4] Learning to reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[5] Beyond Majority Voting: Self-Reflective Test-Time Reinforcement Learning for LLM Reasoning. web. http://www.cse.cuhk.edu.hk/~byu/papers/C357-ICML2026-SR-TTRL.pdf (0000-00-00)
[6] Reward Modeling | RLHF and Post-Training Book. web. https://rlhfbook.com/c/05-reward-models (0000-00-00)
[7] Test-Time Self-Correction for LLM Reasoning. web. https://arxiv.org/html/2608.05643v1 (0000-00-00)
[8] RLHF for Reasoning | The Neural Base. web. https://theneuralbase.com/rlhf-theory/learn/advanced/rlhf-for-reasoning/ (0000-00-00)
[9] ReSPO: Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning. hf-daily. https://huggingface.co/papers/2609.35433 (2026-09-28)
[10] Semifactual Credit-Augmented Policy Optimization. hf-daily. https://huggingface.co/papers/2609.40360 (2026-09-30)
[11] Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning. hf-search. https://huggingface.co/papers/2506.08745 (2025-06-10)
[12] Gains and Collapse in On-Policy Distillation: A Reinforcement Learning Perspective. hf-daily. https://huggingface.co/papers/2610.03185 (2026-10-02)
[13] RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning. hf-search. https://huggingface.co/papers/2505.14140 (2025-05-20)
[14] Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning. hf-search. https://huggingface.co/papers/2512.16917 (2026-03-25)
[15] Likelihood-Based Reward Designs for General LLM Reasoning. hf-search. https://huggingface.co/papers/2602.03979 (2026-02-03)
[16] Reinforcement Learning for LLM Reasoning Under Memory Constraints. hf-search. https://huggingface.co/papers/2504.20834 (2025-04-29)
[17] Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning. hf-search. https://huggingface.co/papers/2505.21178 (2025-05-27)
[18] Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities. hf-search. https://huggingface.co/papers/2602.05281 (2026-02-05)
[19] Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective. hf-search. https://huggingface.co/papers/2506.14965 (2025-06-17)
