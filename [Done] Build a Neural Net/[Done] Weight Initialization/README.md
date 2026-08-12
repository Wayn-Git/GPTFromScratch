# Weight initialization

Lets say we have an neural network or an MLP and we stack like 10-15 layers with random N(0,1) weights you know what will happen ?
The activations will either explode to infinity or collpase to zero killing the overall neuron. This is one of the most common reasons deep networks fail to train and fixing it is pretty simple

**Weight initialization is the technique that makes deep learning practical**

We'll be discussing 3 main weight initialization algorithms

## 1. Random Initialization

Thi is a simple apparoch where we initialize weights randomly within a small range we can use numpy's `np.random.uniform(range, size)` method to simply impliment this

## 2. [Xavier Initialization](https://businessanalytics.substack.com/p/weight-initialization-in-neural-networks)

(Also known as glorot initialization) is an popular technique for initlizing the weights in a neural network the main idea is to set the initial weights of the network in a way that allows the acttivation and gradients to flow effiencetly in both the directions. It considerst he number of input and output units of each layer to determine the scale of the random initlization

### Why are the input and outputs important here ?

They matter because they ensure the signal strenght and gradient remain stable across layers ensuing the values don't explode or vanish

* Forward Control: The number of inputs determines how many values are added together at a neruon more input means the output grows larget so the weight must shrink to keep the activation steady
* Backward pass control: The number of outputs control how gradient flows backward through the network so factoring in both directions balances tghe signal
* This specific initalization method can't be used with ReLU but only with sigmoid
* We calculate the standard deviation of the inputs and the output and then multiply it by a random integer size of the number of input and outputs

## 3. Kaiming Initialization

(Also known as He initialization) is basically the fix for Xavier when you are using ReLU. Remember how I said Xavier only really works with sigmoid? If you try to use Xavier with ReLU it will fail because ReLU zeroes out all the negative numbers, which basically halves the variance of your activations. Kaiming steps in to fix this exact problem so the signal and gradients keep flowing effiencetly.

### Why does ReLU need its own specific initialization ?

They matter because if you just use normal Xavier with a ReLU network, your signal strenght will slowly vanish and die out layer by layer until the network stops learning.

* The Variance Fix: Since ReLU kills off exactly half the signal (all negative values become zero), Kaiming initialization compensates by simply multiplying the variance by 2 to keep things steady.
* Forward Control: Unlike Xavier which looks at both inputs and outputs, Kaiming usually just considers the number of input units (fan-in) to determine the scale. It makes the weights just large enough to survive the ReLU cut as it moves forward.
* This specific initalization method is designed explicitly for ReLU and its variants (like Leaky ReLU), you shouldn't use it for sigmoid or tanh.
* We calculate this by drawing random numbers from a standard normal distribution and then multiplying them by the square root of 2 divided by the number of inputs.