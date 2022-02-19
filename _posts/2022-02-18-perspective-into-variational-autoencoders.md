---
layout: post
title: "Perspective into Variational Autoencoders"
usemathjax: true
---

## Outline
1. Probability distributions
    1. Motivate Gaussian 
    2. Motivate mixture of Gaussians. Talk about the power of latent variables. 
2. Talk about parameter estimation methods for probability distributions - MLE, MAP and Full Bayesian.
    1. Restrict to MLE, illustrate MLE for Gaussian, MLE for mixture of Gaussians(show that there is no closed form analytical solution) by computing derivatives. Mention EM.
3. EM algorithm
    1. Rough outline of EM for gaussian mixtures. Alternative view of EM. Introduce latent variables. 
    2. EM algorithm in general. Intuitive Proof. How to use EM for filling in missing data. 
    3. Mention the singularity problem with MLE and the difficulty to compute the no. of mixtures. Motivate Variational Inference.
4. Explain variational inference - ELBO loss. 
5. Introduce VAE as the marriage of Neural Nets and graphical models. Explain that the VAE is trying to minimize ELBO loss (the KL divergence and squared error terms) and therefore tries to model the input data as a mixture of gaussians, which is powerful than ordinary AE that model data as a gaussian. 
6. Basic Pytorch code for VAE (TF2.0 side by side).


## Probability Basics 

A random variable x denotes a quantity that is uncertain. The variable may denote the result of an experiment 
(e.g., flipping a coin) or a real-world measurement of a fluctuating property (e.g., measuring the temperature). 
If we observe several instances of the measured random variable $\{x_i\}_{i=1}^I$, then it might take a different value 
on each occasion. However, some values may occur more often than others. This information is captured by the probability 
distribution $Pr\left(x\right)$ of the random variable and the choice of the distribution $Pr\left(x\right)$ depends on 
the domain of the data $x$ that it models.

$$

\begin{aligned}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{aligned}
$$
