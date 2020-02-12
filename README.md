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

| title                               |  ipynb    |  slides    |
| ------------------------------ | ---- | ---- |
| Installations with CUDA | ---- | https://d2l.ai/chapter_installation/index.html |
| Basic Operations on GPUs                                    | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb#/) |



### Deep Learning Basic

| title                               |  ipynb    |  slides    |
| ------------------------------ | ---- | ---- |
| Data Manipulation with Ndarray | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb#/) |
| Automatic Differentiation | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb#/) |
| Concise Implementation of Softmax Regression | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb#/) |
| Concise Implementation of Multilayer Perceptron (MLP) | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb#/) |



## Syllabus

In this training, we are going to provide an overview of the in-depth convolutional neural networks (CNN) theory and handy python code. What is more important, the audience would be able to train a simple CNN model on our pre-setup cloud-computing instances for free. Here are the detailed schedule:

1. [Hardware for deep learning](https://d2l.ai/chapter_computational-performance/hardware.html#gpus-and-other-accelerators);
3. Basic convolutional neural networks;
4. Modern convolutional neural networks;
5. TextCNN;
6. AutoML.

### Convolutional Neural Networks

```{.python .input}

```


Notebooks for a 1-day crash course. It aims for teaching deep learning in a single day. This repo contains the notebooks with only simplified code blocks. The texts are also summarized into slides that will be uploaded later.

Check [the wiki page](https://github.com/mli/1day-notebooks/wiki) for instructions to setup the running environments.

## Part 1: Deep Learning Basic

| title                               |  ipynb    |  slides    |
| ------------------------------ | ---- | ---- |
| Data Manipulation with Ndarray | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/1-ndarray.ipynb#/) |
| Automatic Differentiation | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/2-autograd.ipynb#/) |
| Linear Regression Implementation from Scratch | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/3-linear-regression-scratch.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/3-linear-regression-scratch.ipynb#/) |
| Concise Implementation of Linear Regression | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/4-linear-regression-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/4-linear-regression-gluon.ipynb#/) |
| Image Classification Data (Fashion-MNIST) | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/5-fashion-mnist.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/5-fashion-mnist.ipynb#/) |
| Implementation of Softmax Regression from Scratch | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/6-softmax-regression-scratch.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/6-softmax-regression-scratch.ipynb#/) |
| Concise Implementation of Softmax Regression | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/7-softmax-regression-gluon.ipynb#/) |
| Implementation of Multilayer Perceptron from Scratch | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/8-mlp-scratch.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/8-mlp-scratch.ipynb#/) |
| Concise Implementation of Multilayer Perceptron | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-1/9-mlp-gluon.ipynb#/) |

## Part 2: Convolutional Neural Networks

| title                                        | ipynb                                                        | slides                                                         |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GPUs                                         | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/1-use-gpu.ipynb#/) |
| Convolutions                                 | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/2-conv-layer.ipynb#/) |
| Pooling                                      | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/3-pooling.ipynb#/) |
| Convolutional Neural Networks (LeNet)        | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/4-lenet.ipynb#/) |
| Deep Convolutional Neural Networks (AlexNet) | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/5-alexnet.ipynb#/) |
| Networks Using Blocks (VGG)                  | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/6-vgg.ipynb#/) |
| Inception Networks (GoogLeNet)                   | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/7-googlenet.ipynb#/) |
| Residual Networks (ResNet)                   | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-2/8-resnet.ipynb#/) |

## Part 3: Performance

| title                                             | ipynb                                                        | slides                                                         |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| A Hybrid of Imperative and Symbolic Programming   | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/1-hybridize.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/1-hybridize.ipynb#/) |
| Multi-GPU Computation Implementation from Scratch | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/2-multiple-gpus.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/2-multiple-gpus.ipynb#/) |
| Concise Implementation of Multi-GPU Computation   | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/3-multiple-gpus-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/3-multiple-gpus-gluon.ipynb#/) |
| Fine Tuning                                       | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-3/4-fine-tuning.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-3/4-fine-tuning.ipynb#/) |

## Part 4: Recurrent Neural Networks

| title                                                    | ipynb                                                        | slides                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Text Preprocessing                                       | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/1-text-preprocessing.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/1-text-preprocessing.ipynb#/) |
| Implementation of Recurrent Neural Networks from Scratch | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/2-rnn-scratch.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/2-rnn-scratch.ipynb#/) |
| Concise Implementation of Recurrent Neural Networks      | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/3-rnn-gluon.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/3-rnn-gluon.ipynb#/) |
| Gated Recurrent Units (GRU)                              | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/4-gru.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/4-gru.ipynb#/) |
| Long Short Term Memory (LSTM)                            | [github](https://github.com/mli/d2l-1day-notebooks/blob/master/notebooks-4/5-lstm.ipynb) | [nbviewer](https://nbviewer.jupyter.org/format/slides/github/mli/d2l-1day-notebooks/blob/master/notebooks-4/5-lstm.ipynb#/) |

```{.python .input}

```
