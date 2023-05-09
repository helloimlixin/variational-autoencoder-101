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

## Reparameterization Trick

To make the sampling process differentiable, one common approach is the *reparameterization trick*,
$$\mathbf{z}^{(i)} \sim q_\phi (\mathbf{z} | \mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \mu^{(i)}, \sigma^{2(i)}\mathbf{I})$$
$$\mathbf{z}^{(i)} = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
where $\odot$ denotes the Hadamard product.
