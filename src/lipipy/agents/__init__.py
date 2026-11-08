"""Agents, reasoning and the research directions grouped under "AGI".

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

LLM agent patterns
------------------
[ ] Prompting: zero-shot, few-shot, chain-of-thought, self-consistency
[ ] Tree of thoughts, graph of thoughts
[ ] ReAct (reason + act) loop
[ ] Tool / function calling, structured outputs
[ ] Plan-and-execute, task decomposition
[ ] Reflexion, self-critique, self-refine
[ ] Multi-agent debate and orchestration (manager-worker, roles)
[ ] Code-executing agents, sandboxed tools
[ ] Computer-use / browser agents (observation-action loops)

Memory and retrieval
--------------------
[ ] Retrieval-augmented generation (RAG): chunking, dense/sparse retrieval,
    BM25, hybrid search, reranking
[ ] Vector indexes: flat, IVF, HNSW, product quantization
[ ] Episodic and semantic memory stores, summarization memory
[ ] GraphRAG / knowledge-graph retrieval

Reasoning and test-time compute
-------------------------------
[ ] Best-of-N sampling with verifiers
[ ] Process / outcome reward models for search
[ ] MCTS over reasoning steps
[ ] Self-play and self-improvement loops (STaR-style)
[ ] Program synthesis and formal verification (LLM + theorem prover)

Evaluation
----------
[ ] Agent benchmarks harness (task success, cost, steps)
[ ] LLM-as-judge with calibration checks
[ ] Safety evaluations (jailbreak and prompt-injection test sets)

Toward general intelligence (AGI research directions)
-----------------------------------------------------
These are research areas rather than finished algorithms. Entries here are
reference implementations of the core idea at toy scale.

[ ] Universal intelligence measures (Legg-Hutter), AIXI approximations (MC-AIXI)
[ ] Solomonoff induction / minimum description length toy models
[ ] Open-ended learning: POET, novelty search, quality-diversity (MAP-Elites)
[ ] Meta-learning: MAML, Reptile, learning to learn, in-context learning probes
[ ] Continual / lifelong learning: EWC, replay, progressive networks
[ ] World models and predictive processing, JEPA-style objectives
[ ] Active inference and the free-energy principle
[ ] Neuro-symbolic systems (differentiable logic, program induction)
[ ] Cognitive architectures (SOAR, ACT-R, global workspace) as reference designs
[ ] Abstraction and reasoning benchmarks (ARC-style tasks)
[ ] Embodied agents and curriculum learning
[ ] Scalable oversight and alignment: debate, recursive reward modelling,
    weak-to-strong generalization
"""
