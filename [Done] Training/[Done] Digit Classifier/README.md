# Understanding the Digit Classifier Pipeline

Let's break down exactly what each piece of this architecture is doing. We are wiring up this feedforward pipeline:

`Input(784) → Linear(512) → ReLU → Dropout(0.2) → Linear(10) → Sigmoid`

Here is exactly what each component is used for and why it matters:

* **Input(784):** The raw data. Since our handwritten digit images are 28x28 pixels, we can't just feed in a square. We have to flatten that 2D grid into a single 1D array of 784 numbers. Each number simply represents the grayscale intensity of a single pixel.
* **Linear(512):** This is our first hidden layer. It takes those 784 raw pixels and projects them into 512 new features. The network is using these 512 units to look for basic patterns, like edges, loops, or straight lines that make up the shapes of the numbers.
* **ReLU:** We need this to introduce non-linearity. If we just stacked linear layers on top of each other, the math would collapse into one giant linear equation, and the model would be completely useless at learning complex shapes. By turning all the negative numbers to zero, ReLU allows the network to actually learn the complex curves of a handwritten digit.
* **Dropout(0.2):** This is our regularization step. During training, we randomly turn off (or "drop") 20% of the neurons in this layer. Why? Because neural networks are lazy and love to just memorize the training data. By randomly turning off neurons, we force the network to distribute its learning and build robust features instead of relying on just a few specific pixels.
* **Linear(10):** This is the final output layer. We take those 512 learned features and map them down to exactly 10 outputs. These represent the 10 possible classes our image could be (the digits 0 through 9).
* **Sigmoid:** Finally, we pass those 10 raw output numbers through a Sigmoid activation. This squashes the values into a clean range between 0 and 1, turning them into confidence scores. For example, if the 8th output number becomes 0.92, the model is 92% confident that the handwritten image it is looking at is the number 7.