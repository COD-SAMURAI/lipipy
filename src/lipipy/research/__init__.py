"""Minimal, faithful implementations of modern research architectures.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned. Each entry links its paper,
lists any simplifications and reproduces one result from the paper.

Large language models
---------------------
[ ] GPT-2 style decoder (nanoGPT-scale)
[ ] LLaMA-style decoder (RMSNorm, SwiGLU, RoPE, GQA)
[ ] KV cache, speculative decoding
[ ] FlashAttention (tiled / online-softmax attention)
[ ] Sliding-window, sparse and linear attention
[ ] Long-context methods: RoPE scaling (NTK, YaRN), ring attention
[ ] Mixture of experts (top-k routing, load-balancing loss, Switch, DeepSeekMoE)
[ ] Multi-head latent attention (MLA)
[ ] Multi-token prediction
[ ] Scaling-law fitting (Kaplan, Chinchilla)
[ ] Decoding: greedy, beam, temperature, top-k, top-p, min-p

Parameter-efficient fine-tuning and compression
-----------------------------------------------
[ ] LoRA, QLoRA, DoRA
[ ] Adapters, prefix tuning, prompt tuning
[ ] Post-training quantization (GPTQ, AWQ), 1-bit/ternary (BitNet)
[ ] Distillation for LLMs
[ ] Pruning (magnitude, SparseGPT, Wanda)

Alignment and post-training
---------------------------
[ ] Supervised fine-tuning (SFT), instruction tuning
[ ] Reward modelling (Bradley-Terry)
[ ] RLHF with PPO
[ ] DPO, IPO, KTO, ORPO, SimPO
[ ] GRPO and RL with verifiable rewards (RLVR)
[ ] Constitutional AI / RLAIF
[ ] Process reward models

Alternatives to the transformer
-------------------------------
[ ] Structured state-space models: S4, S5
[ ] Mamba, Mamba-2 (selective SSMs)
[ ] RWKV
[ ] RetNet
[ ] Hyena / long convolutions
[ ] xLSTM
[ ] Kolmogorov-Arnold networks (KAN)
[ ] Liquid / continuous-time networks, neural ODEs

Generative modelling
--------------------
[ ] Score-based models, score SDEs
[ ] Classifier and classifier-free guidance
[ ] Latent diffusion (Stable Diffusion style)
[ ] Diffusion transformers (DiT)
[ ] Flow matching, rectified flow
[ ] Consistency models
[ ] Autoregressive image/video generation, visual tokenizers
[ ] Discrete / masked diffusion for text

Multimodal
----------
[ ] Vision-language models (LLaVA-style adapters)
[ ] Perceiver / Perceiver IO
[ ] Speech: CTC, RNN-T, Whisper-style encoder-decoder
[ ] Audio tokenizers / neural codecs

Science and spatiotemporal
--------------------------
[ ] Physics-informed neural networks (PINNs)
[ ] Neural operators (DeepONet, Fourier neural operator)
[ ] Weather / climate forecasters (GraphCast-style mesh GNN)
[ ] Time-series foundation-model patterns (PatchTST, decoder-only forecasters)
[ ] Equivariant networks (E(n)-GNN, SE(3)-transformers)
[ ] Protein-structure building blocks (invariant point attention)

Efficiency and systems
----------------------
[ ] Mixed precision, gradient checkpointing
[ ] Data / tensor / pipeline parallelism (conceptual reference versions)
[ ] Paged attention

Benchmark harness
-----------------
[ ] Shared training loop, config system and seed control
[ ] Standard small benchmarks (e.g. character LM, CIFAR-10, ETT time series)
[ ] Results tables with confidence intervals
"""
