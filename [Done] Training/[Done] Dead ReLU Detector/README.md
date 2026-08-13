# Dead ReLU Detection

Let's understand what exactly a "Dead ReLU" is. We know that the ReLU activation function outputs `max(0, x)`. If a neuron's pre-activation input is negative for *every* sample in the batch, its output becomes exactly 0 for every sample.

Because the output is flat (0), its gradient is also 0. This means the weights will never update during backpropagation, and the neuron effectively "dies" permanently.

## 1. How We Detect It

We detect dead neurons by checking the outputs immediately after each `nn.ReLU` layer.

* We compute the fraction of neurons whose output is exactly 0 for all samples in the batch.
* We return a list of these dead fractions (one per ReLU layer).
* **Point to be noted:** We wrap this detection step in `torch.no_grad()`. This disables gradient calculation, which saves a massive amount of memory since we are just diagnosing the network, not actively training it.

## 2. Suggesting a Fix

Once we have the list of dead fractions per ReLU layer, we decide what to do with them. We suggest the most appropriate fix by checking for these conditions in this exact order:

1. **`'use_leaky_relu'`**: If **any layer has a dead fraction > 0.5**.
* *Why:* This is a severe failure. If half the layer is entirely useless, you need to switch to an activation function that doesn't completely zero out negative values (like Leaky ReLU) to keep the gradients flowing.


2. **`'reinitialize'`**: If the **first layer has a dead fraction > 0.3**.
* *Why:* Early-layer death propagates forward, crippling the rest of the network. This usually means your starting weights were completely off, so you should re-initialize the weights from scratch.


3. **`'reduce_learning_rate'`**: If the dead fractions **strictly increase with depth** AND the **last layer's fraction > 0.1**.
* *Why:* The learning rate is likely too high, aggressively pushing the weights into negative space and causing deeper layers to die off as training progresses.


4. **`'healthy'`**: If the **max dead fraction < 0.1** OR if it simply doesn't match any of the critical failure patterns above.
* *Why:* Having a small percentage of dead neurons (minor dead neurons) is perfectly normal in deep learning and won't stop the model from converging.