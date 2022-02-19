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
If we observe several instances of the measured random variable $$\{x_i\}_{i=1}^I$$, then it might take a different value 
on each occasion. However, some values may occur more often than others. This information is captured by the probability 
distribution $$Pr\left(x\right)$$ of the random variable and the choice of the distribution $$Pr\left(x\right)$$ depends on 
the domain of the data $$x$$ that it models.

A random variable may be discrete or continuous. A discrete random variable takes values from a predefined set. This set 
may be ordered (taking values in ‘Low’, ‘Medium’, ‘High’) or unordered (taking values in ‘Heads’, ‘Tails’). A suitable 
choice of distribution to model a binary outcome, coin tossing experiment where the random variable takes values in 
‘Heads’ or ‘Tails’ could be a bernoulli distribution. A continuous random variable takes values that are real numbers. 
For eg., the height of students in a class, the temperature of a house measured everyday at 3 pm etc. A suitable choice 
of distribution to model these real-valued random variables could be a gaussian distribution. 
 
In machine learning, we describe the measured/collected data from a source as random variables and use probabilities 
over these random variables to account for different kinds of natural variations in the data. Such variations may arise 
due to inherent noise in the measurement process, or due to factors that are unmodeled in the dataset. For e.g., consider 
a dataset having a temperature measurement (which is the random variable) taken at 3 pm everyday in a house. If we do not 
factor in the location at which this reading was taken, there might as well be fluctuations in the temperature depending 
on whether the reading was taken in the bedroom which is air-conditioned or in the kitchen near the stove. Thus, location 
becomes an unmodeled factor that causes fluctuations in the temperature random variable. Therefore, using probability 
distributions over random variables help us account for the unmeasurable noise in the system, and also the measurable 
but un-accounted factors affecting the data collected of the system. Naturally, the data in a machine learning problem 
is always assumed to have been drawn from an underlying, unknown probability distribution. Supervised learning algorithms 
would then be learning to infer a desired quantity, the y’s from the underlying conditional probability distribution 
$$Pr\left(Y\middle| X\right)$$ that it implicitly learns, whereas unsupervised learning algorithms would instead model the 
given data $$Pr\left(X\right)$$ itself to learn useful representations of the data which can then be used for many purposes. 
Before we move onto build up on probability distributions, let us break for a quick side note.
