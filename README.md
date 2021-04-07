# NIN Implementation with Pytorch
- Unofficial implementation of the paper *Network In Network*


## 0. Develop Environment
```
Docker Image
- tensorflow/tensorflow:tensorflow:2.4.0-gpu-jupyter

Library
- Pytorch : Stable (1.7.1) - Linux - Python - CUDA (11.0)
```
- Using Single GPU


## 1. Implementation Details
- model.py : NIN model
- train.py : train NIN
- utils.py : count correct prediction
- NIN - Cifar 10.ipynb : install library, download dataset, preprocessing, train and result
- Visualize - Feature Map.ipynb : visualize the feature map of full activations, top 10% activations
- Details
  * NIN with dropout is hard to train and get the score same as paper, so used NIN without dropout instead
  * Follow the official code train details : batch size 128, momentum 0.9, weight decay 0.00001
  * No learning rate scheduler for convenience
  * No augmentation using global contrast normalization, ZCA whitening
  * Use kaiming normalization for initializing weight parameters of convolution layers
  * Use CIFAR 10 statistics for image pre-processing


## 2. Reference
- Network In Network [[paper]](https://arxiv.org/pdf/1312.4400.pdf) [[official code]](https://github.com/mavenlin/cuda-convnet)
