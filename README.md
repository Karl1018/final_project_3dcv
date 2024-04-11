# Comparative Analysis of Image Colorization Techniques

This repo is the final project of lecture 3dcv.

## Abstract

Automatic image colorization has made significant advancements with the advent of deep learning techniques. Against this background, this project provides a comprehensive comparison of three cutting-edge image colorization methods: Convolutional Neural Networks (CNNs), Generative Adversarial Networks (GANs), and Diffusion Models. We implemented the three mentioned models and delved into the foundational principles of each method, outlining our implementational decisions. 

## Setup

The environment needed can be built via:

```
conda env create -f environment.yml
```

## Training

The train.py contains detailed help list which enables you set various training parameters.

```
python train.py --help
```

## Demo

The demo of our project is available in demo.ipynb.

## Dataset
We use AFHQ cat dataset and use dataset_tool.py to compress the images to 256*256.

The dataset used in our experiments can be downloaded via: 

```
https://drive.google.com/file/d/1qd2fSfJp2-o06S-kJgildS1PBrj353AY/view?usp=sharing
```

