# MNIST Neural Network

We are using the MNIST dataset to create a neural network from scratch.

## Inputs and Outputs

- Each input is a 28×28 image flattened to a vector $x \in \mathbb{R}^{784}$.
- The task is to classify the image into one of 10 classes (digits 0–9).

## Network structure

    0 Layer -> 784 input nodes
    1 Layer -> 10 hidden nodes
    2 Layer -> 10 output nodes

## Forward pass

- Input activations:

$$
A^{(0)} = x
$$

- First (hidden) linear transform and activation:

$$
Z^{(1)} = W^{(1)} A^{(0)} + b^{(1)}
$$

$$
A^{(1)} = \text{ReLU}\bigl(Z^{(1)}\bigr)
$$

applied elementwise. For a scalar $t$:

$$
\text{ReLU}(t) = \max(0,t) =
\begin{cases}
t & t \ge 0,\\
0 & t < 0.
\end{cases}
$$

- Second (output) linear transform and softmax activation:

$$
Z^{(2)} = W^{(2)} A^{(1)} + b^{(2)}
$$

$$
A^{(2)} = \text{softmax}\bigl(Z^{(2)}\bigr)
$$

The softmax components (sigma notation) are:

$$
a^{(2)}_{j} = \frac{e^{Z^{(2)}_{j}}}{\displaystyle\sum_{k=1}^{M} e^{Z^{(2)}_{k}}}, \qquad j=1,\dots,M
$$

where for this network $M=10$. Softmax ensures.

$$
\begin{matrix}
a^{(2)}_{j}>0 \ \ \& \ \
\sum_{j} a^{(2)}_{j} = 1
\end{matrix}
$$

## Parameter shapes (explicit)

- $W^{(1)} \in \mathbb{R}^{10\times 784}$, $b^{(1)} \in \mathbb{R}^{10}$.
- $W^{(2)} \in \mathbb{R}^{10\times 10}$, $b^{(2)} \in \mathbb{R}^{10}$.

## Backward propagation (brief, refactored)

Compute the output error and propagate it backward to form gradients. For a single example:

$$
dZ^{(2)} = A^{(2)} - Y
$$

where $Y$ is the one-hot encoded true label (length $M$). Example for label 4:

$$
Y = \begin{bmatrix}0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\end{bmatrix}
$$

Concise one-hot creation examples:

```python
# list comprehension
y = 4
Y = [1 if i == y else 0 for i in range(10)]
```


