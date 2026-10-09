# Survey of Efficient Inference and Small Language Models

## TL;DR
- Small Language Models (SLMs) leverage compact architectures to achieve high performance with significantly lower memory and latency footprints [1][2][3].
- Model compression techniques such as quantization, structured pruning, and knowledge distillation are essential for optimizing these models for edge deployment [4][5][6].
- Recent benchmarking methodologies, including tinyBenchmarks, enable cost-effective evaluation of SLM reasoning capabilities without sacrificing fidelity [7].
- Speculative decoding has emerged as a primary inference acceleration paradigm, achieving significant speedups by coupling lightweight drafting models with robust verification [8][9][10].
- Advances in architecture-level optimizations, such as lightning attention and model merging, allow for scalable efficiency improvements in modern generative models [11][12].

## Background
Small Language Models (SLMs) are typically defined as transformer-based models with parameter counts ranging from millions to a few billion [13][3]. Unlike massive general-purpose LLMs, SLMs are designed for resource-constrained environments, including mobile devices and local enterprise servers [1][2]. The rise of SLMs is driven by the demand for low-latency inference, reduced operational costs, and the need for data privacy, as these models can often be executed entirely on-device without data leaving the user’s local hardware [1][14][3]. Foundational work in this area includes distillation techniques like those demonstrated in early Fast DistilBERT implementations, which pioneered the co-design of hardware-aware compression and knowledge transfer to smaller architectures [6].

## Methods for Efficient Inference
Model compression remains the primary pathway for creating efficient language models. Knowledge distillation, structured pruning, and quantization allow models to maintain high accuracy while drastically reducing the parameter space [4][5]. For instance, structural pruning approaches like CoFi achieve high speedups by learning compact representations, while Post-Training Quantization (PTQ) has become a standard approach for Mixture-of-Experts (MoE) architectures, allowing researchers to manage large sparse parameter counts without proportional increases in latency [15][5].

More recently, advanced techniques like MC-MoE and D^2MoE have introduced training-free dynamic compressors that utilize mixed-precision and adaptive expert scheduling to optimize memory footprint [4][16]. These approaches are particularly critical for modern MoE models, where effective management of the active parameter count during inference can significantly improve throughput on hardware with restricted memory bandwidth [16]. By combining these techniques, researchers can build SLMs that compete with significantly larger models in specific domains while retaining the ease of deployment characteristic of smaller parameter counts [1][4].

## Architectural Innovations and Model Merging
In addition to compression, recent shifts in model architecture have significantly improved inference scalability. Techniques like lightning attention, introduced in models such as MiniMax-01, demonstrate how hybrid MoE architectures can leverage parallel compute strategies for efficient long-context processing [11]. Innovations in recurrent and state-space models have also introduced gating mechanisms—as seen in Gated Delta Networks—which improve performance through more efficient update rules [17].

Furthermore, research into model merging has revolutionized how researchers explore the model landscape without incurring the high costs of training. By recycling and merging existing model checkpoints, researchers can identify Pareto-optimal configurations that avoid the usual accuracy-latency trade-offs [12]. This approach, combined with decoding-aware pruning methods like SparseDecoding—which accounts for self-generation token dynamics—provides a holistic path toward deploying compact yet highly capable models [18].

## Benchmarking and Performance Metrics
Assessing the performance of SLMs requires a departure from standard, resource-intensive evaluation frameworks. Current research highlights that model effectiveness is highly task-dependent, complicating the use of monolithic benchmarks [19]. In response, new methodologies like tinyBenchmarks have been proposed to enable evaluation using high-quality, reduced subsets of datasets like MMLU, providing comparable rankings to full-scale evaluations at a fraction of the compute cost [7].

Furthermore, researchers are increasingly prioritizing hardware-centric metrics such as tokens-per-second, power consumption, and first-token latency [1][20]. The H2O-Danube3 series represents this shift, focusing on architectures optimized specifically for the constraints of mobile environments [20]. As summarized by recent comprehensive reviews, the industry is moving away from purely theoretical FLOPS-based efficiency metrics toward end-to-end performance measures that capture the real-world deployment trade-offs between model accuracy and system latency [21].

## Trends and Open Problems
The most significant trend in the last two years is the shift toward hardware-aware inference acceleration, particularly through advancements in speculative decoding (SD) [8][22][9]. SD improves decoding speed by employing a small "drafting" model to propose token sequences that are then verified in parallel by the target model [8]. While promising, current deployments face challenges in draft effectiveness, particularly on small models (1-2B parameters) where the drafting overhead often negates potential speed gains [10]. 

Emerging research into Distributed Split Speculative Decoding (DSSD) is addressing these communication bottlenecks by partitioning the verification logic between the edge device and central servers, achieving notable speedup gains [9]. Despite these gains, several open problems persist: the efficient integration of speculative decoding into multimodal architectures remains an active area of dispute, as language-specific drafters often struggle with vision-language tasks [23]. Additionally, the lack of standardized benchmarks for agentic workloads—where model performance is measured by task success rather than token generation speed—remains a major gap for developers of the next generation of SLMs [8].

## References
[1] A Survey of Small Language Models. web. https://arxiv.org/html/2410.20011v1 (2024-10-25)
[2] Small Language Models: Survey, Measurements, and Insights. web. https://arxiv.org/html/2409.15790v1 (2024-09-24)
[3] Small but Mighty: A Comparative Review of Small Language Models and Their Advantages. web. https://link.springer.com/chapter/10.1007/978-981-95-0629-3_22 (2024-01-01)
[4] MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More. hf-search. https://huggingface.co/papers/2410.06270 (2024-10-08)
[5] Structured Pruning Learns Compact and Accurate Models. hf-search. https://huggingface.co/papers/2204.00408 (2022-04-01)
[6] Fast DistilBERT on CPUs. hf-search. https://huggingface.co/papers/2211.07715 (2022-10-27)
[7] tinyBenchmarks: evaluating LLMs with fewer examples. hf-search. https://huggingface.co/papers/2402.14992 (2024-02-22)
[8] Towards Optimal Speculative Decoding: A Comprehensive Survey on Optimizations and Future Directions. web. https://dl.acm.org/doi/10.1145/3846171 (2026-09-16)
[9] DSSD: Efficient Edge-Device Deployment and Collaborative Inference via Distributed Split Speculative Decoding. web. https://arxiv.org/html/2507.12000v1 (2025-01-01)
[10] An Empirical Study of Speculative Decoding for Small Language Models. web. https://aclanthology.org/2026.eacl-long.255.pdf (2026-01-01)
[11] MiniMax-01: Scaling Foundation Models with Lightning Attention. hf-daily. https://huggingface.co/papers/2501.08313 (2025-01-14)
[12] If You Can't Use Them, Recycle Them: Optimizing Merging at Scale Mitigates Performance Tradeoffs. hf-daily. https://huggingface.co/papers/2412.04144 (2024-12-05)
[13] What are Small Language Models (SLM)?. web. https://www.ibm.com/think/topics/small-language-models (2024-11-05)
[14] SLMs vs LLMs: What are small language models?. web. https://www.redhat.com/en/topics/ai/llm-vs-slm (2024-01-01)
[15] QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts. hf-search. https://huggingface.co/papers/2406.08155 (2024-06-12)
[16] D^2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving. hf-search. https://huggingface.co/papers/2504.15299 (2025-04-17)
[17] Gated Delta Networks: Improving Mamba2 with Delta Rule. hf-daily. https://huggingface.co/papers/2412.06464 (2024-12-09)
[18] SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference. hf-daily. https://huggingface.co/papers/2610.12327 (2026-10-08)
[19] EfficientLLM: Efficiency in Large Language Models. hf-search. https://huggingface.co/papers/2505.13840 (2025-05-20)
[20] H2O-Danube3 Technical Report. hf-search. https://huggingface.co/papers/2407.09276 (2024-07-12)
[21] A Survey on Efficient Inference for Large Language Models. hf-search. https://huggingface.co/papers/2404.14294 (2024-04-22)
[22] Efficient Inference for Edge Large Language Models: A Survey. web. https://www.sciopen.com/local/article_pdf/10.26599/TST.2025.9010166.pdf (2025-01-01)
[23] Speculative Decoding: How It Evolved, When It Stays Lossless, and What's Next. web. https://neurips2026-speculative-decoding.vercel.app/ (2026-01-01)
