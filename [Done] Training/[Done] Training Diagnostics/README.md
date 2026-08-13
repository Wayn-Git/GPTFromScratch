# Training Diagostics

This is where we diagnose our training loop to make sure it works the difference between a working GPt and a broken one is often a single initialization mistake.

Training diagnostics is the skill that catches those issues by inspecting

- Acitvations
- Gradients
- Loss curves

to figure out why a model isn't learning 

## 1. Activation Stats (The Forward Signal)

**What it is used for:** Tracking the actual numbers coming out of each layer as data moves forward through the network.

**Why we check it:**

* **Mean & Standard Deviation (std):** We need to know if the data signal is remaining stable as it passes through the layers. If the `std` drops below **0.1**, the signal is fading away. If the `std` shoots above **10.0**, the numbers are blowing up out of control.
* **Dead Fraction:** When using activation functions like ReLU, negative numbers become zero. If a neuron outputs **<= 0** for *every* single sample in a batch, it is considered "dead." A dead neuron passes no information forward and receives no learning updates backward. We track this to make sure large chunks of our network aren't paralyzed and useless.

## 2. Gradient Stats (The Learning Signal)

**What it is used for:** Tracking the size of the updates being passed backward through the network during backpropagation.

**Why we check it:**

* **Mean & Standard Deviation:** Helps us see if the learning updates are balanced or heavily skewed in one direction.
* **L2 Norm (The overall size):** This is the most critical metric for gradients because it tells us exactly how big the weight update is going to be.
* If the norm is **> 1000**, our updates are too massive. The model will destabilize and jump wildly past the correct answer (**exploding gradients**).
* If the norm is **< 1e-5** (especially in the last layer), our updates are microscopic. The weights won't change, and the model will freeze and stop learning altogether (**vanishing gradients**).



## 3. The Diagnosis (Automating the Fix)

**What it is used for:** Automatically flagging the exact reason the network is failing so we know how to fix it (like changing the initialization or lowering the learning rate).

**Why we check in this specific order:**

1. **Dead Neurons (> 50%):** If more than half the layer is dead, the network is fundamentally broken. This usually means the learning rate was so high that it forced the neurons into negative space, permanently killing them.
2. **Exploding Gradients:** This is the next most catastrophic failure. If updates are massively huge (norm > 1000 or activation std > 10.0), the model is unstable and cannot learn.
3. **Vanishing Gradients:** If the network isn't dead and isn't exploding, we check if the learning signal has simply faded away (norm < 1e-5 or activation std < 0.1).
4. **Healthy:** If the network passes all of these checks, our signals are flowing efficiently in both directions, and the model is actively learning!