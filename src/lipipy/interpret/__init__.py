"""Interpretability, explainability and model analysis.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Model-agnostic explanations
---------------------------
[ ] Permutation feature importance
[ ] Partial dependence plots, individual conditional expectation (ICE)
[ ] Accumulated local effects (ALE)
[ ] LIME
[ ] SHAP (Kernel SHAP, Tree SHAP, exact Shapley values for small models)
[ ] Counterfactual explanations, anchors

Gradient-based attribution
--------------------------
[ ] Saliency maps (vanilla gradients)
[ ] Gradient x input, SmoothGrad
[ ] Integrated gradients
[ ] Grad-CAM, Grad-CAM++
[ ] Layer-wise relevance propagation (LRP), DeepLIFT

Representation analysis
-----------------------
[ ] Linear probes
[ ] Representation similarity: CKA, SVCCA, RSA
[ ] Concept activation vectors (TCAV)
[ ] Logit lens, tuned lens

Mechanistic interpretability
----------------------------
[ ] Activation patching / causal tracing
[ ] Path patching, attribution patching
[ ] Circuit discovery (induction heads, IOI-style circuits)
[ ] Sparse autoencoders and transcoders for feature discovery
[ ] Steering vectors / activation addition
[ ] Superposition toy models

Attention analysis
------------------
[ ] Attention maps, attention rollout, attention flow

Data-centric analysis
---------------------
[ ] Influence functions
[ ] Data Shapley, TracIn
[ ] Training-dynamics maps (dataset cartography)

Robustness, fairness and uncertainty
------------------------------------
[ ] Adversarial examples: FGSM, PGD, C&W
[ ] Certified robustness (randomized smoothing)
[ ] Fairness metrics (demographic parity, equalized odds) and mitigations
[ ] Uncertainty: MC dropout, deep ensembles, conformal prediction
[ ] Calibration metrics (ECE, reliability diagrams)
"""
