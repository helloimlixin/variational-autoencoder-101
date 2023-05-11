# variational-autoencoder-101
Some notes and experiments on Variational Autoencoders

References: 
1. [From Autoencoder to Beta-VAE by Lil'Log](https://lilianweng.github.io/posts/2018-08-12-vae/)
2. [Getting Started with Variational Autoencoders using PyTorch by Sovit Ranjan Rath](https://debuggercafe.com/getting-started-with-variational-autoencoders-using-pytorch/)
3. [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)

## Overview

Unlike traditional autoencoders, instead of mapping the input to a fixed vector, we map it into a *distribution* in the Variational Autoencoders (VAE). Denote the data distribution as $p_\theta$ with parameterization $\theta$, $\mathbf{z}$ as the latent encoding vector, and $\mathbf{x}$ as the input data, we have,

- Prior $p_\theta(\mathbf{z})$
- Likelihood $p_\theta (\mathbf{x} | \mathbf{z})$
- Posterior $p_\theta (\mathbf{z} | \mathbf{x})$

Denote the real parameter for this distribution as $\theta^\ast$, which can be thought of as the parameter that maximizes the log likelihood of generating **real** data samples:
$$\theta^\ast = \arg \max_\theta \sum_{i = 1}^n \log p_\theta (\mathbf{x}^{(i)}),$$
which follows the sampling process,

1. Sample a latent vector $\mathbf{z}^{(i)}$ from the prior $p_{\theta^\ast} (\mathbf{z})$.
2. Generate a value $\mathbf{x}^{(i)}$ from a conditional distribution $p_{\theta^\ast} (\mathbf{x} | \mathbf{z} = \mathbf{z}^{(i)})$.

The data generation then follows,
$$p_\theta (\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}|\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z},$$

which is not tractable and often we would compute an approximation function $q_\phi (\mathbf{z}|\mathbf{x}^{(i)})$ to approximate the posterior $p_\theta (\mathbf{z}|\mathbf{x}^{(i)})$, parameterized by $\phi$. To minimize the distance between the two probabilities, we minimize the Kullback-Leibler divergence with respect to $\phi$,
$$\min_\phi D_{KL} (q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) | p_\theta(\mathbf{z}|\mathbf{x}^{(i)})),$$
which can be expanded as,
$$
\begin{aligned}
D_{KL} &(q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta(\mathbf{z}|\mathbf{x}^{(i)})) \\
&= \int q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \log \frac{q_\phi (\mathbf{z}|\mathbf{x}^{(i)})}{p_\theta (\mathbf{z}|\mathbf{x}^{(i)})} d\mathbf{z} \\
&= \int q_\phi(\mathbf{z}|\mathbf{x}^{(i)})\log \frac{q_\phi(\mathbf{z}|\mathbf{x}^{(i)})p_\theta(\mathbf{x}^{(i)})}{p_\theta(\mathbf{z}, \mathbf{x}^{(i)})} d \mathbf{z} \\
&= \int q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \left(\log p_\theta(\mathbf{x}^{(i)}) + \log \frac{q_\phi(\mathbf{z}|\mathbf{x}^{(i)})}{p_\theta(\mathbf{z}, \mathbf{x}^{(i)})}\right) d\mathbf{z} \\
&= \log p_\theta (\mathbf{x}^{(i)}) + \int q_\phi (\mathbf{z}|\mathbf{x}^{(i)}) \log \frac{q_\phi (\mathbf{z}|\mathbf{x}^{(i)})}{p_\theta(\mathbf{z}, \mathbf{x}^{(i)})} d\mathbf{z}\quad (\text{because } \int q(\mathbf{z} | \mathbf{x}^{(i)}) \mathbf{z} = 1) \\
&= \log p_\theta (\mathbf{x}^{(i)}) + \int q_\phi (\mathbf{z} | \mathbf{x}^{(i)}) \log \frac{q_\phi (\mathbf{z} | \mathbf{x}^{(i)})}{p_\theta (\mathbf{x}^{(i)} | \mathbf{z}) p_\theta (\mathbf{z})} d\mathbf{z} \\
&= \log p_\theta (\mathbf{x}^{(i)}) + \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x}^{(i)})} \left[ \log \frac{q_\phi (\mathbf{z}|\mathbf{x}^{(i)})}{p_\theta (\mathbf{z})} - \log p_\theta (\mathbf{x}^{(i)}|\mathbf{z}) \right] \\
&= \log p_\theta (\mathbf{x}^{(i)}) + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta (\mathbf{z})) - \mathbb{E}_{\mathbf{z} \sim q_\phi (\mathbf{z} | \mathbf{x}^{(i)})} \log p_\theta (\mathbf{x}^{(i)}|\mathbf{z}),
\end{aligned}
$$
which can be expressed as,
$$
\log p_\theta (\mathbf{x}^{(i)}) = D_{KL} (q_\phi (\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta (\mathbf{z} | \mathbf{x}^{(i)})) + \mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}),
$$
where $\mathcal{L}(\theta, \phi; \mathbf{x}^{(i)})$ is called the *variational lower bound*,
$$
\mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) = - D_{KL} (q_\phi (\mathbf{z}|\mathbf{x}^{(i)}) \| p_\theta(\mathbf{z})) + \mathbb{E}_{\mathbf{z} \sim q_\phi (\mathbf{z} | \mathbf{x}^{(i)})} \left[ \log p_\theta (\mathbf{x}^{(i)}|\mathbf{z}) \right]
$$

## Solution of the KL Term, Gaussian Case

Here's the solution to the KL term with both the prior $p_\theta (\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and the posterior approximation $q_\phi(\mathbf{z}|\mathbf{x}^{(i)})$ are Gaussian. Let $J$ be the dimensionality of $\mathbf{z}$.
Let $\boldsymbol{\mu}^{(i)}$ and $\mathbf{\sigma}^{(i)}$ denote the variational mean and standard deviation vectors at datapoint $i$, which are outputs of the encoding MLP, i.e., nonlinear functions of datapoint $\mathbf{x}^{(i)}$ and the variational parameters $\phi$, and let $\mu^{(i)}_j$ and $\sigma^{(i)}_j$ denote the $j$-th element of these vectors, then we have,
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

## Reparameterization Trick

To make the sampling process differentiable, one common approach is the *reparameterization trick*, which is an alternative way for generating samples from the posterior $q_\phi (\mathbf{z}|\mathbf{x})$. Let $\mathbf{z}$ be a continuous random variable and $\mathbf{z} \sim q_\phi (\mathbf{z}|\mathbf{x})$ be some conditional distribution. We can express the random variable $\mathbf{z}$ via a *deterministic* mapping $\mathbf{z} = g_\phi (\boldsymbol{\epsilon}, \mathbf{x})$, where $\boldsymbol{\epsilon}$ is an auxiliary variable with an independent marginal distribution $p(\boldsymbol{\epsilon})$, and $g_\phi (\cdot)$ is some vector-valued function parameterized by $\phi$. We can then construct a *differentiable estimator*:
$$
\int q_\phi (\mathbf{z}|\mathbf{x}) f(\mathbf{z}) d \mathbf{z} \approx \frac{1}{L} \sum_{l=1}^L f(g_\phi (\mathbf{x}, \boldsymbol{\epsilon}^{(l)})),
$$
where $\boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})$, $L$ is the number of sampes per datapoint drawn, which can be set to $1$ as long as the minibatch size $M$ is large enough (e.g., $M=100$, shown experimentally as mentioned in the paper), where the minibatch $\mathbf{X}^M = \{ \mathbf{x}^{(i)} \}_{i = 1}^M$ is a randomly drawn sample of $M$ datapoints from the full dataset $\mathbf{X}$ with $N$ datapoints.

Hence we can sample from the posterior $\mathbf{z}^{(i,l)} \sim q_\phi (\mathbf{z} | \mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \mu^{(i)}, (\sigma^{(i)})^2\mathbf{I})$ using,
$$\mathbf{z}^{(i, l)} = g_\phi (\mathbf{x}^{(i)}, \boldsymbol{\epsilon}^{(l)}) = \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\epsilon}^{(l)}, \text{ where } \boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
where $\odot$ denotes the Hadamard product.

## Variational Auto-Encoder

With the loss function and the reparameterization trick explained above, we have the resulting estimator for the Variational Auto-Encoder model and datapoint $\mathbf{x}^{(i)}$ is,
$$
\mathcal{L}(\theta, \phi; \mathbf{x}^{(i)}) \approx \frac{1}{2} \sum_{j = 1}^J \left(1 + \log ((\sigma_j^{(i)})^2) - (\mu_j^{(i)})^2 - (\sigma_j^{(i)})^2 \right) + \frac{1}{L} \sum_{l = 1}^L \log p_\theta (\mathbf{x}^{(i)}|\mathbf{z}^{(i, l)})
$$
$$
\text{where } \mathbf{z}^{(i, l)} = \boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\epsilon}^{(l)}, \text{ and } \boldsymbol{\epsilon}^{(l)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

For continuous data, the decoding term $\log p_\theta (\mathbf{x}^{(i)}|\mathbf{z}^{(i, l)})$ is a Gaussian MLP as explained below.

## Gaussian MLPs as Encoder or Decoder

Here we let the encoder or decoder be a multivariate Gaussian with a diagonal covariance structure,

### Encoder

When using MLPs as encoders, we have for a given datapoint $\mathbf{x}^{(i)} \in \mathbb{R}^{n_1 \times n_2}$,
$$
\begin{aligned}
    \log q_\phi (\mathbf{z}^{(i)}|\mathbf{x}^{(i)}) &= \log \mathcal{N}(\mathbf{z}^{(i)}; \boldsymbol{\mu}^{(i)}, (\boldsymbol{\sigma}^{(i)})^2 \mathbf{I}) \\
    \text{where } \boldsymbol{\mu}^{(i)} &= \mathbf{W}_2 \mathbf{h}^{(i)} + \mathbf{b}_2 \\
    \log (\boldsymbol{\sigma}^{(i)})^2 &= \mathbf{W}_3 \mathbf{h}^{(i)} + \mathbf{b}_3 \\
    \mathbf{h}^{(i)} &= \mathrm{ReLU} (\mathbf{W}_1 \mathbf{x}^{(i)} + \mathbf{b}_1)
\end{aligned}
$$
where as mentioned above, the mean and the standard deviation $\boldsymbol{\mu}^{(i)}$ and $\boldsymbol{\sigma}^{(i)}$, are the outputs of the encoding MLP, i.e., nonlinear functions of datapoint $\mathbf{x}^{(i)}$ and $\{\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3\}$ are the weights and biases of the MLP encoder and are the variational parameters $\phi$.

Take the MNIST dataset for example, we have the flattened datapoint vector $\mathbf{x}^{(i)} \in \mathbb{R}^{784}$ ($28 \times 28 \times 1 = 784$), if we set the output channel dimension of the first encoder layer as $512$, we have $h^{(i)} \in \mathbb{R}^{512}$, then if we set the number of hidden features as $16$ (aka. latent dimension), we have the output from the second encoder layer is a length-$2 \times 16 = 32$ vector, consisting of the mean and the log-variance vectors that are both $16$ in length. The resulting latent vector is then of shape $\mathbf{z}^{(i)} \in \mathbb{R}^{16}$.

### Decoder

Let the decoder be a multivariate Gaussian with a diagonal covariance structure, denote the dimension of the output image as $n_1 \times n_2$,
$$
\begin{aligned}
    \log p (\mathbf{x}^{(i)}|\mathbf{z}^{(i, l)}) &= \mathcal{N} (\mathbf{x}^{(i)}; \boldsymbol{\mu}^{(i, l)}, (\boldsymbol{\sigma}^{(i, l)})^2\mathbf{I}) \\
    \text{where } \boldsymbol{\mu}^{(i, l)} &= \mathbf{W}_5 \mathbf{h}^{(i, l)} + \mathbf{b}_5 \\
    \log (\boldsymbol{\sigma}^{(i, l)})^2 &= \mathbf{W}_6 \mathbf{h}^{(i, l)} + \mathbf{b}_6 \\
    \mathbf{h}^{(i, l)} &= \mathrm{ReLU}(\mathbf{W}_4 \mathbf{z}^{(i, l)} + \mathbf{b}_4)
\end{aligned}
$$
where $\{\mathbf{W}_4, \mathbf{W}_5, \mathbf{W}_6, \mathbf{b}_4, \mathbf{b}_5, \mathbf{b}_6\}$ are the weights and biases of the MLP and part of $\theta$ when the MLP is used as an decoder.
