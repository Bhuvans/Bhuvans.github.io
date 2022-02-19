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

The following are the key components that need to be defined when trying to apply any ML technique. Clearly putting down 
these components should set us up for solving the actual problem.
+ Data: Sample of example data points collected from the source. For a supervised learning task, this could be the 
$$\{x_i,y_i\}_{i=1}^n$$ pairs collected from a data source. 
+ Model: Parameterized functional approximation of the relation between the data and the target quantity of interest. 
For a simple neural network, say just a single sigmoidal neuron, this could take the form of $$\hat{y}=\frac{1}{1+e^{-\left(w^Tx\right)}}$$
+ Parameters: Unknown variables/parameters of the model that need to be learned. For the above neural network model, this 
would be the w’s.
+ Learning algorithm: An algorithm for learning the parameters of the model. For the above neural network model, this 
could be the stochastic gradient descent or a variant algorithm like Adam, RMSprop etc. 
+ Objective/loss function: Some sort of a cost that evaluates the model’s predictions. The purpose of the loss function is 
to guide the learning algorithm to achieve the task. For the above neural network model, assuming it is used for regression, 
the loss could be a squared error loss. On the other hand, Cross entropy could be a suitable loss function for a 
classification task.
+ Hyperparameters: They can be thought to control the complexity of the model learned. Ideally, these need to be tuned 
for each new dataset before a ‘good’ model of sufficient complexity  can be trained. In the context of a general neural 
network model, this could be the number of hidden layers in the network, the number of hidden units per layer, the 
learning rate used in the learning algorithm etc.

Going back to random variables and probability distributions, we saw that describing data with probability distributions 
provides a mathematical framework to express all forms of uncertainty and noise associated with the model. Machine learning, 
or learning from data, in most cases, therefore primarily concerns with fitting probability distribution models to the 
observed data $$\{x_i\}_{i=1}^I$$ by making certain assumptions . This process is referred to as learning because we learn 
about the parameters $$\theta$$ of the model. More generally, these are also called parameter estimation methods since the 
goal is to estimate parameter values $$\theta$$ that best explain the given observations $$x$$. A related concern of 
parameter estimation is that of calculating the probability of a new datum $$x^\ast$$ under the resulting model. This is 
known as evaluating the learned model or the predictive distribution. We now consider three methods of parameter estimation: 
maximum likelihood, maximum a posteriori and the full bayesian approach.   

Let us now consider a probabilistic model defined by random variable(s) X and parameter(s) $$\theta$$. Bayes’ theorem 
says that, $$P\left(\theta\middle|X\right)=\frac{P\left(X\middle|\theta\right).P\left(\theta\right)}{P\left(X\right)}$$. 
Here, $$P\left(\theta\middle| X\right)$$ is the posterior, $$P\left(\theta\right)$$ is the prior and 
$$P\left(X\middle|\theta\right)$$ is the likelihood function . In parameter estimation, the goal is to find the parameters 
$$\theta$$ of the model that best explains the given data $$X$$ or that $$\theta$$ which gives the maximum value for the 
probability $$P\left(\theta\middle| X\right)$$ given $$X$$.

