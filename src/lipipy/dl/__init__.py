"""Deep learning building blocks. Requires ``pip install lipipy[dl]``.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned. Each item is checked against
PyTorch (or the original paper) before it is ticked.

Foundations
-----------
[ ] Scalar and tensor autograd engine (reverse-mode automatic differentiation)
[ ] Computational graph, backpropagation
[ ] Parameter initialization: Xavier/Glorot, He/Kaiming, orthogonal, LSUV

Layers
------
[ ] Linear / dense, bias
[ ] Embedding
[ ] Convolution 1D/2D/3D, transposed, dilated, depthwise-separable, grouped
[ ] Pooling: max, average, global, adaptive
[ ] Recurrent: vanilla RNN, LSTM, GRU, bidirectional, stacked
[ ] Attention: additive (Bahdanau), dot-product (Luong), scaled dot-product,
    multi-head, cross-attention, multi-query (MQA), grouped-query (GQA)
[ ] Normalization: batch, layer, instance, group, RMSNorm, weight norm
[ ] Regularization: dropout, DropConnect, stochastic depth, label smoothing
[ ] Residual and skip connections, highway layers, gating (GLU, SwiGLU)

Activations
-----------
[ ] Sigmoid, tanh, ReLU, Leaky ReLU, PReLU, ELU, SELU, GELU, SiLU/Swish,
    Mish, Softplus, Softmax, LogSoftmax, Hard-swish

Losses
------
[ ] MSE, MAE, Huber / smooth L1
[ ] Binary cross-entropy, categorical cross-entropy, focal loss
[ ] Hinge, squared hinge
[ ] KL divergence, negative log-likelihood
[ ] Contrastive, triplet, InfoNCE / NT-Xent
[ ] CTC loss
[ ] Dice / IoU losses (segmentation)

Optimizers and schedules
------------------------
[ ] SGD, momentum, Nesterov
[ ] Adagrad, RMSProp, Adadelta
[ ] Adam, AdamW, NAdam, RAdam, AMSGrad
[ ] LAMB, LARS, Adafactor
[ ] Lion, Sophia, Shampoo, Muon
[ ] Learning-rate schedules: step, exponential, cosine, warmup, one-cycle,
    warmup-stable-decay (WSD)
[ ] Gradient clipping, gradient accumulation, mixed precision

Architectures: vision
---------------------
[ ] Multilayer perceptron (MLP)
[ ] LeNet, AlexNet, VGG
[ ] GoogLeNet / Inception
[ ] ResNet, ResNeXt, Wide ResNet, DenseNet
[ ] MobileNet, ShuffleNet, EfficientNet, ConvNeXt
[ ] U-Net, FCN, DeepLab (segmentation)
[ ] R-CNN family, YOLO, SSD, RetinaNet, DETR (detection)
[ ] Vision Transformer (ViT), DeiT, Swin Transformer
[ ] MLP-Mixer

Architectures: sequence and language
------------------------------------
[ ] Seq2seq with attention
[ ] Transformer (encoder-decoder, "Attention Is All You Need")
[ ] Encoder-only (BERT-style, masked language modelling)
[ ] Decoder-only (GPT-style, causal language modelling)
[ ] Positional encodings: sinusoidal, learned, relative, RoPE, ALiBi
[ ] Temporal convolutional networks (TCN), WaveNet
[ ] Tokenizers: BPE, WordPiece, Unigram

Architectures: graphs
---------------------
[ ] Graph convolutional network (GCN)
[ ] GraphSAGE
[ ] Graph attention network (GAT)
[ ] Graph isomorphism network (GIN)
[ ] Message passing neural networks (MPNN)
[ ] Spatiotemporal GNNs (ST-GCN, DCRNN, Graph WaveNet)

Representation and self-supervised learning
-------------------------------------------
[ ] Autoencoders: undercomplete, denoising, sparse, contractive
[ ] Word2Vec (skip-gram, CBOW), GloVe
[ ] Siamese networks
[ ] SimCLR, MoCo, BYOL, SimSiam, Barlow Twins, DINO
[ ] Masked autoencoders (MAE)
[ ] CLIP (contrastive image-text)
[ ] JEPA (joint-embedding predictive architectures)

Generative models (base versions; modern variants in lipipy.research)
---------------------------------------------------------------------
[ ] Restricted Boltzmann machine, deep belief network
[ ] Variational autoencoder (VAE), beta-VAE, VQ-VAE
[ ] GAN, DCGAN, conditional GAN, WGAN / WGAN-GP
[ ] Normalizing flows (RealNVP, Glow)
[ ] PixelCNN / autoregressive image models
[ ] Denoising diffusion (DDPM, DDIM)

Training techniques
-------------------
[ ] Data augmentation (flips, crops, mixup, cutmix, RandAugment)
[ ] Transfer learning and fine-tuning
[ ] Knowledge distillation
[ ] Early stopping, checkpointing
[ ] Pruning, quantization
"""
