# MNIST Neural Network

We are using the MNIST dataset to create a neural network from scratch.

## Inputs and Outputs

- Each input is a 28×28 image flattened to a vector $\mathbf{x}\in\mathbb{R}^{784}$.
- The task is to classify the image into one of 10 classes (digits 0–9).

## Network structure

    0 Layer -> 784 input nodes
    1 Layer -> 10 hidden nodes
    2 Layer -> 10 output nodes

## Forward Pass

- Input activations:

  $$\mathbf{A}^{(0)} = \mathbf{x}. $$

- First (hidden) linear transform and activation:

  $$\mathbf{Z}_{(1)} = W_{(1)}\mathbf{A}_{(0)} + \mathbf{b}_{(1)},$$
  $$\mathbf{A}^{(1)} = \mathrm{ReLU}(\mathbf{Z}^{(1)})$$

  applied elementwise. For a scalar $t$:

  $$\mathrm{ReLU}(t) = \max(0,t) = \begin{cases} t & t\ge 0, \\
  0 & t<0. \end{cases}$$

- Second (output) linear transform and softmax activation:

  $$\mathbf{Z}_{(2)} = W_{(2)}\mathbf{A}_{(1)} + \mathbf{b}_{(2)},$$
  $$\mathbf{A}_{(2)} = \mathrm{softmax}(\mathbf{Z}_{(2)}).$$

  The softmax components (using sigma notation) are:

  $$a^{(2)}_j = \frac{e^{Z^{(2)}_j}}{\Sigma_{j=1}^{K} e^{Z^{(2)}_j}},\qquad j=1,\dots,M,$$

  where for this network $M=10$. Softmax ensures $a{^j}_{(2)}>0$ and $\sum_{j} a{^j}_{(2)} = 1$.

## Parameter shapes (explicit)

- $W^{(1)}\in\mathbb{R}^{10\times 784}$, $\mathbf{b}^{(1)}\in\mathbb{R}^{10}$.
- $W^{(2)}\in\mathbb{R}^{10\times 10}$, $\mathbf{b}^{(2)}\in\mathbb{R}^{10}$.
e0r4

## Backward Propagation
Improving the weights and biases by compiling respectful error that contributed to the absolute error in the predictions.

$$\mathbf{{dZ}_{(2)}} = \mathbf{{A}_{(2)}} - Y$$

Here Y is the actual value i.e. 

- If actual value corresponds to be 4 it will be encoded as 
  
$$ Y = \begin{bmatrix} 0&0&0&0&1&0&0&0&0&0 \end{bmatrix} $$

As shown the encoding uses the following code:
```python
y = 4
Y = []
for i in range(10):
  if i == y:
    Y.append(1)
  else:
    Y.append(0)
```
The time complexity of this code will be:
$$ \mathbb{O}\mathbf(N)$$

As the code follows it is quite a bit inefficient but since this is only computed once every training it will not matter but if need be we can make it efficient we can just convert the list to a hash of predefined size and fill it up at the initialization process so we will just be having a time complexity of: $$ \mathbb{O}\mathbf(1) $$

