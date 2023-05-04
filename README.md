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

which is not tractable and often we would compute an approximation function $q_\phi (\mathbf{z}|\mathbf{x})$ to approximate the posterior $p_\theta (\mathbf{z}|\mathbf{x})$, parameterized by $\phi$. To minimize the distance between the two probabilities, we minimize the Kullback-Leibler divergence with respect to $\phi$,
$$\min_\phi D_{KL} (q_\phi(\mathbf{z}|\mathbf{x}) | p_\theta(\mathbf{z}|\mathbf{x})),$$
which can be expanded as,
$$
\begin{aligned}
D_{KL} &(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) \\
&= \int q_\phi(\mathbf{z}|\mathbf{x}) \log \frac{q_\phi (\mathbf{z}|\mathbf{x})}{p_\theta (\mathbf{z}|\mathbf{x})} d\mathbf{z} \\
&= \int q_\phi(\mathbf{z}|\mathbf{x})\log \frac{q_\phi(\mathbf{z}|\mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d \mathbf{z} \\
&= \int q_\phi(\mathbf{z}|\mathbf{x}) \left(\log p_\theta(\mathbf{x}) + \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})}\right) d\mathbf{z} \\
&= \log p_\theta (\mathbf{x}) + \int q_\phi (\mathbf{z}|\mathbf{x}) \log \frac{q_\phi (\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z}\quad (\text{because } \int q(\mathbf{z} | \mathbf{x})d \mathbf{z} = 1) \\
&= \log p_\theta (\mathbf{x}) + \int q_\phi (\mathbf{z} | \mathbf{x}) \log \frac{q_\phi (\mathbf{z} | \mathbf{x})}{p_\theta (\mathbf{x} | \mathbf{z}) p_\theta (\mathbf{z})} d\mathbf{z} \\
&= \log p_\theta (\mathbf{x}) + \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{q_\phi (\mathbf{z}|\mathbf{x})}{p_\theta (\mathbf{z})} - \log p_\theta (\mathbf{x}|\mathbf{z}) \right] \\
&= \log p_\theta (\mathbf{x}) + D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta (\mathbf{z})) - \mathbb{E}_{\mathbf{z} \sim q_\phi (\mathbf{z} | \mathbf{x})} \log p_\theta (\mathbf{x}|\mathbf{z}),
\end{aligned}
$$
which can be expressed as,
$$
\log p_\theta (\mathbf{x}) = D_{KL} (q_\phi (\mathbf{z}|\mathbf{x}) \| p_\theta (\mathbf{z} | \mathbf{x})) + \mathcal{L}(\theta, \phi; \mathbf{x}),
$$
where $\mathcal{L}(\theta, \phi; \mathbf{x})$ is called the *variational lower bound*,
$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = - D_{KL} (q_\phi (\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z})) + \mathbb{E}_{\mathbf{z} \sim q_\phi (\mathbf{z} | \mathbf{x})} \left[ \log p_\theta (\mathbf{x}|\mathbf{z}) \right]
$$

## Reparameterization Trick
