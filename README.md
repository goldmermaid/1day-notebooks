# GTC2020 Tutorial - Dive into Deep Learning

Last updated：|today|

Deep learning is transforming the world nowadays. However, realizing deep learning presents unique challenges because any single application brings together various disciplines. Applying deep learning requires simultaneously understanding:

1. the engineering required to train models efficiently, navigating the pitfalls of numerical computing and getting the most out of available hardware;
2. the mathematics of a given modeling approach;
3. the optimization algorithms for fitting the models to data;
4. and the experience of choosing proper hyperparameters for the solution.


To fulfill the strong wishes of simpler but more practical deep learning materials, [Dive into Deep Learning](https://d2l.ai/), a unified resource of deep learning was born to achieve the following goals:

- Offering depth theory and runnable code, showing readers how to solve problems in practice;
- Allow for rapid updates, both by us, and also by the community at large;
- Be complemented by a forum for interactive discussions of technical details and to answer questions;
- Be freely available for everyone.



## Prerequisites


### Installations

- [Installations with CUDA](https://d2l.ai/chapter_installation/index.html)
- [Basic Operations on GPUs](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb#/) ]



### Deep Learning Basics

| title                               |  notes    |  slides    |
| ------------------------------ | ---- | ---- |
| Data Manipulation with Ndarray | [D2L](https://d2l.ai/chapter_preliminaries/ndarray.html) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb#/) |
| Automatic Differentiation | [D2L](https://d2l.ai/chapter_preliminaries/autograd.html) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb#/) |
| Softmax Regression | [D2L](https://d2l.ai/chapter_linear-networks/softmax-regression.html) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb#/) |
| Concise Implementation of Multilayer Perceptron (MLP) | [D2L](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb#/) |



## Syllabus

In this training, we are going to provide an overview of the in-depth convolutional neural networks (CNN) theory and handy python code. What is more important, the audience would be able to train a simple CNN model on our pre-setup cloud-computing instances for free. Here are the detailed schedule:


| Time | Topics |
| --- | --- |
| 9:00---9:10 | [Deep Learning Introduction](#Deep-Learning-Introduction) |
| 9:10---9:30 | [Convolutional Neural Networks](#Convolutional-Neural-Networks) |
| 9:30---9:40 | [Overview of NLP](#Overview-of-NLP) |
| 9:40---10:10 | [TextCNN on Sentiment Analysis](#TextCNN-on-Sentiment-Analysis) |
| 10:10---10:20 | [AutoML] |
| 10:30---10:45 | [Resources and Q&A](#Resources-and-Q&A ) | 



### Deep Learning Introduction

| title                                        | ipynb                                                        | slides                                                         |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Hardware for deep learning |  | [Notes](https://d2l.ai/chapter_computational-performance/hardware.html#gpus-and-other-accelerators) |



### Convolutional Neural Networks
| title                                        | ipynb                                                        | slides                                                         |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Convolutions                                 | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb#/) |
| Pooling                                      | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb#/) |
| Convolutional Neural Networks (LeNet)        | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb#/) |
| Deep Convolutional Neural Networks (AlexNet) | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb#/) |
| Networks Using Blocks (VGG)                  | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb#/) |
| Inception Networks (GoogLeNet)                   | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb#/) |
| Residual Networks (ResNet)                   | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb#/) |

### Overview of NLP

1. NLP Roadmap
1. Downstream Tasks
1. Models
1. Word Embedding : GloVe, etc.



### TextCNN on Sentiment Analysis



### AutoML


### Resources and Q&A 


```{.python .input}

```

```{.python .input}

```
