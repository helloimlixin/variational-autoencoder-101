# Linear Variational Autoencoder

This is the implementation of the linear VAE model using PyTorch.

## Usage

Specify the training parameters by command line arguments:

```bash
python train.py -e 20 -b 64 -lr 0.0001
```

## Solution of the KL Term, Gaussian Case

Here's the solution to the KL term with both the prior $p_\theta (\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and the posterior approximation $q_\phi(\mathbf{z}|\mathbf{x}^{(i)})$ are Gaussian. Let $J$ be the dimensionality of $\mathbf{z}$.
Let $\boldsymbol{\mu}^{(i)}$ and $\mathbf{\sigma}^{(i)}$ denote the variational mean and standard deviation vectors at datapoint $i$, which are outputs of the encoding MLP, i.e., nonlinear functions of datapoint $\mathbf{x}^{(i)}$ and the variational parameters $\phi$, and let $\mu^{(i)}_j$ and $\sigma^{(i)}_j$ denote the $j$-th element of these
vectors, then we have,
$$
\begin{aligned}
\int q_\theta (\mathbf{z}|\mathbf{x}^{(i)}) \log p_\theta (\mathbf{z}) d\mathbf{z} &= \int \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, (\boldsymbol{\sigma}^{(i)})^2) \log \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I}) d\mathbf{z} \\
&= - \frac{J}{2} \log (2\pi) - \frac{1}{2} \sum_{j = 1}^J \left((\mu_j^{(i)})^2 + (\sigma_j^{(i)})^2\right),
\end{aligned}
$$
and,
$$
\begin{aligned}
\int q_\theta (\mathbf{z}|\mathbf{x}^{(i)}) \log q_\theta (\mathbf{z}|\mathbf{x}^{(i)}) d \mathbf{z} &= \int \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, (\boldsymbol{\sigma}^{(i)})^2) \log \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, (\boldsymbol{\sigma}^{(i)})^2) d \mathbf{z} \\
&= - \frac{J}{2} \log (2\pi) - \frac{1}{2} \sum_{j = 1}^J (1 + \log (\sigma^{(i)}_j)^2),
\end{aligned}
$$
therefore,
$$
\begin{aligned}
-D_{KL} (q_\phi (\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta (\mathbf{z})) &= \int q_\phi (\mathbf{z}|\mathbf{x}^{(i)}) (\log p_\theta (\mathbf{z}) - \log q_\phi (\mathbf{z}|\mathbf{x}^{(i)})) d \mathbf{z} \\
&= \frac{1}{2} \sum_{j = 1}^J (1 + \log ((\sigma^{(i)}_j)^2) - (\mu^{(i)}_j)^2 - (\sigma^{(i)}_j)^2),
\end{aligned}
$$
this result is then used in the loss function computation in the implementation.