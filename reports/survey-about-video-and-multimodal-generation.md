# Survey of Video and Multimodal Generation

## TL;DR
- Video generation has evolved from GANs to diffusion-based architectures and Diffusion Transformers (DiTs), achieving unprecedented levels of physical realism and controllability [1][2].
- State-of-the-art generation utilizes training-free motion control and temporal attention mechanisms, such as MC-Sparse, to manage long-horizon sequences efficiently [3][4][5].
- Evaluation has shifted from pixel-level heuristics to multi-dimensional, human-aligned benchmarks, emphasizing spatiotemporal consistency and text-video alignment [6][7][8].
- Open problems include enhancing temporal coherence in long-form generation, mitigating hallucinations in automated evaluation, and developing physically grounded world models [9][8][10].

## Background
Video generation involves synthesizing sequential visual data to represent motion and scene evolution, a field that has witnessed a paradigm shift in the last five years. Early foundations were dominated by Generative Adversarial Networks (GANs), which provided foundational approaches for frame-level generation but struggled with long-term temporal coherence [1]. The rise of diffusion models in 2022 transformed the field, offering superior visual fidelity and controllability through iterative denoising [2][11].

More recently, the adoption of Transformer-based backbones, specifically Diffusion Transformers (DiTs), has enabled the scaling of models to massive datasets, leading to the emergence of "world simulator" paradigms [1][9]. These models are designed to simulate complex physical interactions, marking a transition toward more intelligent, physically grounded video synthesis [1][2]. The maturation of this field is crucial for future applications in interactive media, autonomous simulation, and immersive experiences [6].

## Architectural Paradigms and Evolution
The evolution of video generation architectures is characterized by a departure from early 3D U-Nets toward massive Transformer architectures. While GANs were once the gold standard, their instability in generating long, consistent video streams led to the dominance of Diffusion Models [1][11]. Modern approaches, such as Autoregressive (AR) diffusion models, bridge the gap between sequential causal modeling—common in Large Language Models (LLMs)—and the iterative denoising processes of traditional diffusion [12][11].

Current trends highlight the transition to DiT backbones, which allow for better scalability and performance in high-dimensional spatiotemporal spaces [9]. Post-training alignment, including supervised fine-tuning and reward-based optimization, is increasingly used to address challenges like error accumulation, which was historically a limitation in purely generative models [9].

## Multimodal Control and Temporal Consistency
Achieving high-quality motion control without retraining has become a central objective in recent research. Training-free frameworks such as MotionClone and BroadWay leverage motion cloning and temporal self-guidance, respectively, to enhance text-video alignment and motion fidelity [3][4]. These approaches avoid the computational burden of fine-tuning, making high-quality control accessible for various diffusion-based models [3][13].

Furthermore, temporal consistency is being improved through architectural innovations like the Cross-frame Textual Guidance Mechanism (CTGM) in FancyVideo [14]. For long-sequence video generation, efficient attention mechanisms are vital; the MC-Sparse framework addresses the performance gap in sparse attention methods for diffusion transformers, optimizing both quality and computational cost for long-form video synthesis [5].

## Evaluation Benchmarks and Challenges
The evaluation of video generation has moved beyond simplistic metrics like Fréchet Video Distance (FVD) or Inception Score (IS), which are now considered insufficient for modern, multi-aspect models [7]. Recent efforts, such as EvalCrafter and Video-Bench, advocate for evaluation frameworks that incorporate four pillars: video quality, text-video alignment, motion quality, and temporal consistency [7][8].

These modern benchmarks utilize "chain-of-query" and few-shot scoring mechanisms with Multimodal Large Language Models (MLLMs) to achieve higher correlation with human perception [8]. However, challenges remain, particularly regarding the inherent ambiguity in automated scoring and the difficulty of evaluating long-form physical grounding [8][10]. Ensuring safety and mitigating hallucination in automated evaluation remain critical areas of ongoing research [6][10].

## Trends and Open Problems
A significant trend in the last two years is the move toward "world models" that aim for physically grounded simulation rather than just pixel synthesis [1]. The integration of audio-visual modalities and the development of open frameworks, such as Wan 2.1, are actively shaping the landscape of accessible video generation [1].

Despite this progress, several problems remain open. Long-horizon temporal consistency remains elusive, as models often suffer from drift or loss of semantic coherence over extended durations [9][5]. Furthermore, the lack of standardized, human-aligned, and robust evaluation protocols means that current metrics still struggle to fully capture subjective human preference in complex, multi-modal tasks [8][10]. Bridging these gaps is essential for the next generation of video synthesis technology.

## References
[1] Evolution of Video Generative Foundations. web. https://arxiv.org/html/2604.06339v1 (2026-01-01)
[2] Video diffusion generation: comprehensive review and open problems. web. https://link.springer.com/article/10.1007/s10462-025-11331-6 (2025-08-20)
[3] MotionClone: Training-Free Motion Cloning for Controllable Video Generation. hf-search. https://huggingface.co/papers/2406.05338 (2024-06-08)
[4] BroadWay: Boost Your Text-to-Video Generation Model in a Training-free Way. hf-search. https://huggingface.co/papers/2410.06241 (2024-10-08)
[5] MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap in Diffusion Transformers. hf-daily. https://huggingface.co/papers/2610.06801 (2026-10-05)
[6] Generative AI Video Evaluation: Survey of Metrics, Benchmarks, and Trustworthiness. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/html/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.html (2026-06-01)
[7] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. web. https://arxiv.org/abs/2310.11440v3 (2024-03-23)
[8] Video-Bench: Human-Aligned Video Generation Benchmark. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Video-Bench_Human-Aligned_Video_Generation_Benchmark_CVPR_2025_paper.pdf (2025-06-01)
[9] Video Generation Models: A Survey of Post-Training and Alignment. web. https://arxiv.org/html/2610.00812 (2026-09-30)
[10] VideoLLM Benchmarks and Evaluation: A Survey. web. https://arxiv.org/abs/2505.03829 (2025-05-01)
[11] Survey of Video Diffusion Models: Foundations, Implementations, and Applications. web. https://arxiv.org/html/2504.16081v1 (2025-04-16)
[12] Spatiotemporal Consistency in Video Generation. web. https://dl.acm.org/doi/10.1145/3802588 (2026-05-18)
[13] Motion-Zero: Zero-Shot Moving Object Control Framework for Diffusion-Based Video Generation. hf-search. https://huggingface.co/papers/2401.10150 (2024-01-18)
[14] FancyVideo: Towards Dynamic and Consistent Video Generation via Cross-frame Textual Guidance. hf-search. https://huggingface.co/papers/2408.08189 (2024-08-15)
