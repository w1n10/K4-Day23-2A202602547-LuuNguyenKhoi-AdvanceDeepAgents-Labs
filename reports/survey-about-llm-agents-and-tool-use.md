# A Survey of LLM-based Autonomous Agents and Tool Use

## TL;DR
- LLM agents leverage reasoning and tool-use paradigms to bridge internal knowledge with external capabilities [1][2][3].
- Systematic design patterns, including multi-agent cooperation, enhance agentic reliability and planning efficacy [4][5].
- Benchmarking is shifting towards evaluating long-horizon oversight, epistemic humility, and predictive spatial reasoning [6][7][8].
- Reliability, safety, and risk-awareness remain significant open challenges despite advances in safety-constitution frameworks [9][10][11].

## Background
LLM-based autonomous agents are systems defined by their ability to perceive, reason, and act in environments to achieve goals [12][13]. Unlike standard language models that produce text, agents are designed to interact with external tools and APIs, transforming static models into dynamic, task-oriented systems [1][14]. This field has matured significantly through foundational research on reasoning-acting loops, such as the ReAct framework [3], which interleaves reasoning traces and task-specific actions to guide tool interaction [3]. The LOMAR architecture (LLM, Objective, Memory, Action, Rethink) further conceptualizes agents into integrated subsystems to manage complexity, memory, and reflection [2]. This evolution is driven by the necessity for agents to operate in high-stakes domains—such as science, engineering, and finance—where reliable execution is mandatory [12].

## Paradigms and Architectural Design
Agentic methods are organized into a hierarchy of design patterns that define how agents decompose tasks and utilize tools. Foundational work, such as Toolformer, demonstrated that models could learn to trigger external API calls as special tokens during generation [14]. This has evolved into more structured, system-theoretic frameworks that categorize agentic systems into functional subsystems—including Perception, Reasoning, Action Execution, and Adaptation [4]. Advanced planning methods now focus on task decomposition and iterative plan selection, aiming to mitigate the limitations of monolithic LLM reasoning [15].

Recent studies indicate that separating planning and reasoning agents significantly boosts performance on multi-step tasks [5]. In these cooperative frameworks, planning agents coordinate the high-level strategy, while reasoning agents execute the fine-grained logical steps [5]. Furthermore, empirical studies on persistent context, such as coding "READMEs" for agents, show that these artifacts act as crucial configuration files that anchor an agent's reasoning process during long-horizon projects [16]. These findings demonstrate that structured agentic workflows are essential for moving beyond ad-hoc designs.

## Evaluation Frameworks and Benchmarks
The landscape of agent benchmarking has moved beyond simple task-success metrics toward evaluating specific reasoning and behavioral traits. Learn2Play Bench assesses whether agents learn from experience in novel environments, contrasting this with simple knowledge retrieval [17]. Similarly, SpaceCast-Bench evaluates predictive spatial reasoning by forcing models to anticipate unseen outcomes, highlighting a gap in static models' dynamic understanding [8].

Evaluating agents in long-horizon tasks requires more sophisticated methods, such as AgentMonBench, which provides evidence-grounded oversight of autonomous decisions [7]. Additionally, the "Identify, Solve, and Escalate" (ISE) framework measures epistemic humility—the ability of an agent to recognize and communicate uncertainty when retrieved evidence contradicts prior knowledge [6]. Finally, frameworks like TestPrism are redefining coding benchmarks to prevent reliance on single reference solutions, requiring test suites to effectively validate both correct and incorrect candidate implementations [18].

## Security, Safety, and Multi-Agent Collaboration
Reliability remains the foremost challenge for high-stakes deployment. Research shows that agentic systems are prone to indirect prompt injection and internal vulnerabilities [19][20]. The AGENT-SAFETYBENCH project highlighted that most popular LLM agents score below 60% in safety metrics, primarily due to a lack of robustness and insufficient awareness of the consequences of their actions [10]. To address these risks, the "Agent-Constitution" framework enforces safety through planning-aware strategies that monitor agent actions at multiple stages—pre-, in-, and post-planning [9].

Multi-agent systems (MAS) offer a way to distribute decision-making but introduce significant communication overhead [21][22]. Recent work proposes the use of "mediator models" to manage the necessity and costs of agent interactions [22]. Furthermore, the ToolEmu framework demonstrates that real-world agent failure modes can be reliably replicated in sandboxed environments, facilitating scalable safety evaluation [11]. Despite these advancements, the field continues to grapple with the conflict between maintaining helpful agent autonomy and enforcing rigid safety constraints in dynamic environments [21][11].

## Trends and Open Problems
The last two years have marked a transition from ad-hoc agent experiments to systematic, architecture-driven development. A major unresolved issue is the reliability gap: current benchmarks demonstrate that agents often prioritize task success over acknowledging their limitations, as evidenced by deficiencies in epistemic humility and risk awareness [6][10]. Furthermore, the persistence of "shortcut learning," where agents rely on superficial patterns instead of deep reasoning, poses a risk to their generalization in novel scenarios [21]. Future research is increasingly focused on developing formal verification, simulation-based testing, and robust "Agent-Constitutions" to bridge the gap between impressive lab performance and dependable real-world autonomous deployment [9][11].

## References
[1] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use, Planning (Including RAG), and Feedback Learning. web. https://aclanthology.org/2025.coling-main.652.pdf (2025-01-01)
[2] From language to action: a review of large language models as autonomous agents and tool users. web. https://link.springer.com/article/10.1007/s10462-025-11471-9 (2026-01-06)
[3] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2022-10-07)
[4] Agentic Design Patterns: A System-Theoretic Framework. hf-search. https://huggingface.co/papers/2601.19752 (2026-01-27)
[5] Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models. hf-search. https://huggingface.co/papers/2410.20007 (2024-10-25)
[6] Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict. hf-daily. https://huggingface.co/papers/2610.12360 (2026-10-08)
[7] What Did the Agent Actually Do? Evidence-Grounded Oversight for Long-Horizon Agents. hf-daily. https://huggingface.co/papers/2610.06406 (2026-10-05)
[8] SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models. hf-daily. https://huggingface.co/papers/2610.12402 (2026-10-08)
[9] TrustAgent: Towards Safe and Trustworthy LLM-based Agents. web. https://aclanthology.org/2024.findings-emnlp.585.pdf (2024-11-01)
[10] AGENT-SAFETYBENCH: Evaluating the Safety of LLM Agents. web. https://www.arxiv.org/pdf/2412.14470v1 (2024-12-19)
[11] IDENTIFYING THE RISKS OF AGENTS WITH AN LM-EMULATED SANDBOX (ToolEmu). web. https://proceedings.iclr.cc/paper_files/paper/2024/file/7274ed909a312d4d869cc328ad1c5f04-Paper-Conference.pdf (2024-05-07)
[12] A survey on large language model based autonomous agents. web. https://dl.acm.org/doi/10.1007/s11704-024-40231-1 (2024-03-22)
[13] Agentic Large Language Models, a Survey. web. https://www.jair.org/index.php/jair/article/download/18675/27253 (2024-01-01)
[14] Toolformer: language models can teach themselves to use tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[15] Understanding the planning of LLM agents: A survey. hf-search. https://huggingface.co/papers/2402.02716 (2024-02-05)
[16] Agentic Coding Tools: An Empirical Study of Context Files. hf-search. https://huggingface.co/papers/2511.12884 (2025-11-17)
[17] Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?. hf-daily. https://huggingface.co/papers/2610.08215 (2026-10-08)
[18] TestPrism: Rethinking Test Evaluation Beyond a Single Reference. hf-daily. https://huggingface.co/papers/2610.12289 (2026-10-08)
[19] Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In. hf-search. https://huggingface.co/papers/2410.16950 (2024-10-22)
[20] Navigating the Risks: A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents. web. https://arxiv.org/abs/2411.09523 (2024-11-14)
[21] A Survey on LLM-based Multi-agent Systems. web. https://link.springer.com/article/10.1007/s44336-024-00009-2 (2024-10-08)
[22] Intelligent Agents in Multi-Agent Systems. web. https://arxiv.org/pdf/2401.03428 (2024-01-08)
