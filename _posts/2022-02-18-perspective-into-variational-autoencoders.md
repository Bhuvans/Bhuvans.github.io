---
layout: post
title: "Perspective into Variational Autoencoders"
katex: True
---

## Outline
1. Probability distributions
    1. Motivate Gaussian. 
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


### Probability Basics 

A random variable $x$ denotes a quantity that is uncertain. The variable may denote the result of an experiment 
(e.g., flipping a coin) or a real-world measurement of a fluctuating property (e.g., temperature readings). If we observe 
several instances of the measured random variable $$\{x_i\}_{i=1}^I$$, then it might take a different value on each occasion. 
However, some values may occur more often than others. This information is captured by the probability distribution 
$Pr\left(x\right)$ of the random variable. 

The choice of the distribution $Pr\left(x\right)$ depends on the domain of the data $x$ that it models. The random 
variable may be discrete or continuous. A discrete random variable takes values from a predefined set. This set may be 
ordered (taking values in ‘Low’, ‘Medium’, ‘High’) or unordered (taking values in ‘Heads’, ‘Tails’). A suitable choice of 
distribution to model a coin tossing experiment with binary outcomes could be a bernoulli distribution. The corresponding 
random variable is discrete-valued and takes values in 'Heads’ or ‘Tails’. A continuous random variable takes values that 
are real numbers. For eg., the height of students in a class, the temperature of a house measured everyday at 3 pm etc. 
A suitable choice of distribution to model these real-valued random variables could be a gaussian distribution. 
 
In machine learning, we describe the measured or observed data from a source as random variables and use probabilities 
over these random variables to account for different kinds of natural variations in the data. The variations may arise 
due to inherent noise in the measurement process, or due to factors that are unmodeled in the dataset. For e.g., consider 
a dataset having a temperature measurement(which is the random variable) taken at 3 pm everyday in a house. If we do not 
factor in the location at which this reading was taken, there might as well be fluctuations in the temperature depending 
on whether the reading was taken in the bedroom which is air-conditioned or in the kitchen near the hot stove. Here, location 
becomes an unmodeled factor that causes fluctuations in the temperature random variable. Therefore, using probability 
distributions over random variables help us account for the unmeasurable noise in the system, and also the measurable 
but unaccounted factors affecting the data collected of the system. Naturally, the data in a machine learning problem 
is always assumed to have been drawn from an underlying, unknown probability distribution. Supervised learning algorithms 
would then be learning to infer a desired quantity, the $y’s$ from the underlying conditional probability distribution 
$Pr\left(Y\middle| X\right)$ that it implicitly learns, whereas unsupervised learning algorithms would instead learn to 
model the given data $Pr\left(X\right)$ itself to derive useful representations of the data which can then be used for 
many purposes. Before we move onto build up on learning these probability distributions, let us break for a quick side note.

The following are the key components that need to be defined when trying to apply any ML technique. Clearly putting down 
these components should set us up for solving the actual problem.
+ ##### Data: 
Sample of example data points collected from the source. For a supervised learning task, this could be the 
$\{x_i,y_i\}_{i=1}^n$ pairs collected from a data source. 
+ ##### Model: 
Parameterized functional approximation of the relation between the data and the target quantity of interest. For a simple 
neural network, say just a single sigmoidal neuron, this could take the form of $\hat{y}=\frac{1}{1+e^{-\left(w^Tx\right)}}$
+ ##### Parameters: 
Unknown variables/parameters of the model that need to be learned. For the above neural network model, this 
would be the $w’s$.
+ ##### Learning algorithm: 
An algorithm for learning the parameters of the model. For the neural network model, this could be the stochastic 
gradient descent or a variant algorithm like Adam, RMSprop etc. 
+ ##### Objective/loss function: 
Some sort of a cost that evaluates the model’s predictions. The purpose of the loss function is to guide the learning 
algorithm to achieve the task. For the neural network model, assuming it is used for regression, the loss could be 
a squared error loss. On the other hand, Cross entropy could be a suitable loss function for a classification task.
+ ##### Hyperparameters: 
They can be thought to control the complexity of the model learned. Ideally, these need to be tuned for each new dataset 
before a ‘good’ model of sufficient complexity  can be trained. In the context of a general neural network model, this 
could be the number of hidden layers in the network, the number of hidden units per layer, the learning rate used in the 
learning algorithm etc.

Going back to random variables and probability distributions, we saw that describing data with probability distributions 
provides a mathematical framework to express all forms of uncertainty and noise associated with the model. Machine learning, 
or learning from data, in most cases, therefore primarily concerns with fitting probability distribution models to the 
observed data $$\{x_i\}_{i=1}^I$$ by making certain assumptions. This process is referred to as learning because we learn 
about the parameters $\theta$ of the model. More generally, these are also called **parameter estimation** methods since the 
goal is to estimate parameter values $\theta$ of the probability distribution that best explain the given observations $x$ 
under an assumed form of the data distribution. A related concern of parameter estimation is that of calculating the 
probability of a new datum $x^\ast$ under the resulting model. This is known as evaluating the learned model or the 
predictive distribution. 

We will now look at three methods of parameter estimation, namely the maximum likelihood estimation, 
maximum a posteriori estimation and and the full bayesian approach. Consider a probabilistic model defined by random variable(s) $X$ and parameter(s) $\theta$. Bayes’ theorem 
says that, $P\left(\theta\middle|X\right)=\frac{P\left(X\middle|\theta\right).P\left(\theta\right)}{P\left(X\right)}$. 
Here, $P\left(\theta\middle| X\right)$ is the posterior, $P\left(\theta\right)$ is the prior and 
$P\left(X\middle|\theta\right)$ is the likelihood. In parameter estimation, the goal is to find the parameters 
$\theta$ of the model that best explains the given data $X$. That is to find that $\theta$ which gives the maximum value for the 
probability $P\left(\theta\middle| X\right)$ given $X$.

Using the Bayes’ theorem, three types of parameter estimation can be done:
##### 1. Maximum-Likelihood Estimation: 
In the Bayes’ therorem equation, we can consider $P\left(X\right)$ a constant quantity that does not figure in the optimization 
with respect to $\theta$ since $X$ is fixed and given. Also, if the prior is non-informative there is no information about 
what $\theta$ is the best to start with, in which case the quantity $P\left(\theta\right)$ is also irrelevant in the optimization 
because it would be the same for all the $\theta$. Therefore, to maximize $P\left(\theta\middle| X\right)$ all that must be 
done is to maximize the likelihood, $P\left(X\middle|\theta\right)$. If we can assume that the data samples were generated 
independently, then we can write $L\left(\theta\middle| X\right)=P\left(X\middle|\theta\right)=\prod_{x\in X}{P\left(x\middle|\theta\right)}$. 
Taking log on both sides gives us the log-likelihood, $l\left(\theta\right)=logL\left(\theta\middle| X\right)=\sum_{x\in X} log P\left(x\middle|\theta\right)$. 
This also converts the pesky product terms to manageable sum of log terms over all the data samples. The log-likelihood is the 
quantity that is then maximized wrt $\theta$ to give the maximum likelihood parameter estimate. 
$$ \widehat{\theta_{ML}}=argmax_\theta\sum_{x\in X} log P\left(x\middle|\theta\right) $$
where $\hat{\theta}_{ML}$ is a single, point estimate for the parameter $\theta$ that is then used to evaluate the probability of 
a new datapoint $\widetilde{x}$ given the training data $X$ by $p\left(\widetilde{x}\middle| X\right)=p\left(\widetilde{x}\middle|\widehat{\theta_{ML}}\right)$.

Let us consider a simple coin-toss experiment as an example. Let $C$ denote the random variable that the coin turns head. 
An outcome, $c = 1$ for the random variable $C$ therefore denotes a head on the coin and an outcome $c = 0$ denotes a 
tail on the coin. Let $\rho$ denote the probability of getting heads on tossing the coin. Choosing a Bernoulli distribution to 
model this simple coin-toss experiment gives the likelihood below, 
$$ p\left(C=c\middle|\rho\right) &= \rho^c\left(1-\rho\right)^{1-c}  
Log-likelihood,  l\left(\rho\right) &= \sum_{i=1}^{N}logp\left(C=c_i\middle|\rho\right)  
 &= n^{\left(1\right)}log\rho+n^{\left(0\right)}log\left(1-\rho\right) $$
where $N$ is the number of data points or the number of independent coin tosses performed, $n^{\left(1\right)}$ is the number 
of heads observed in the $N$ data points and $n^{\left(0\right)}$ is the number of tails observed in the $N$ data points. 
Maximizing the log-likelihood implies $\frac{\partial l}{\partial\rho}=0$ which gives $\widehat{\rho_{ML}}=\frac{n^{\left(1\right)}}{n^{\left(1\right)}+n^{\left(0\right)}}$
as the maximum likelihood estimate for the parameter $\rho$ assuming that the tosses of the coin can be modelled using a bernoulli 
distribution.

##### 2. Maximum a-posteriori: 
In the maximum likelihood estimation method, we did not have any information about the parameters or the 
prior distribution $P\left(\theta\right)$. Now we consider the case when do know something about the parameters $\theta$. 
For eg., what could be known about the parameter $\rho$ in the coin-toss experiment? Say, we know that the coin is fair with a 
high probability. Think of $\rho$ being a gaussian with a peak at $\rho = 0.5$. Is that all? No, we are still not done. 
A gaussian is a bad choice for the parameter $\rho$ which only takes values in $\[0,1\]$. A better choice for the 
distribution of $\rho$ would be a beta distribution that spans only over the $\[0, 1\]$ domain. The probability distribution 
for a beta distribution is as follows: $Beta\left(\rho\middle|\alpha,\beta\right)=P\left(\rho\middle|\alpha,\beta\right)=\frac{1}{B\left(\alpha,\beta\right)}\rho^{\alpha-1}\left(1-\rho\right)^{\beta-1}$. 
Here $B\left(\alpha,\beta\right)$ is the beta function defined by $\frac{\Gamma\left(\alpha+\beta\right)}{\Gamma\left(\alpha\right)\Gamma\left(\beta\right)}=\frac{1}{B\left(\alpha,\beta\right)}$. 
For eg. $Beta\left(\rho\middle|1,\ 1\right)$ reduces to a uniform distribution or a non-informative prior. An informative prior 
to encode a fair coin in our example could be denoted by say, $Beta\left(\rho\middle|5,\ 5\right)$. Note that the $\alpha$ 
and $\beta$ parameters of the beta distribution act like pseudo-counts of the outcomes, ie. the Heads and Tails of the coin toss 
experiment. 

Let us now write down the defining statement for the maximum a-posteriori estimate as $\widehat{\theta_{MAP}}=argmax_\theta p\left(\theta\middle| X\right)=argmax_\theta p\left(X\middle|\theta\right)p\left(\theta\right)$. 
Note that like in the case of maximum likelihood estimation we have avoided computing the entire posterior $p\left(\theta\middle| x\right)$ to find the 
parameter $\theta$ that maximizes the posterior. This is because the denominator $P\left(X\right)$ is a constant when maximizing 
the posterior with respect to $\theta$. After taking log and summing the quantity over all the datapoints, this is equivalent to 
$$\widehat{\theta_{MAP}}=argmax_\theta\{\sum_{x\in X} log p\left(x\middle|\theta\right)+log\ p\left(\theta\right)\}$$. For the example 
coin toss experiment, this would take the form of $$argmax_\theta\left\[n^{\left(1\right)}log\rho+n^{\left(0\right)}log\left(1-\rho\right)+\left(\alpha-1\right)log\rho+\left(\beta-1\right)log\left(1-\rho\right)\ +\ CONST.\right\]$$
Like in the maximum likelihood case, taking the derivative of this objective and equating to zero gives $$\widehat{\rho_{MAP}}=\frac{n^{\left(1\right)}+\alpha-1}{n^{\left(1\right)}+n^{\left(0\right)}+\alpha+\beta-2}$$. 
$${\hat{\theta}}_{MAP}$$ is also a single, point estimate for the parameter $$\theta$$ that is then used to evaluate the probability 
of a new datapoint $$\widetilde{x}$$ given the training data $$X$$ by $$p\left(\widetilde{x}\middle| X\right)=p\left(\widetilde{x}\middle|\widehat{\theta_{MAP}}\right)$$.

##### 3. Full Bayesian: 
With maximum likelihood and maximum a posteriori estimation, we found point estimates of the parameter, 
$$\widehat{\theta_{ML}}$$ and $$\widehat{\theta_{MAP}}$$ respectively, that give a single best parameter setting that best explains 
the data that is given. However, in terms of finding the best prediction for a new data point ie. $$p\left(\widetilde{x}\middle| X\right)$$, 
using point estimates for $$\theta$$ are not very accurate. Consider the defining equation for $$p\left(\widetilde{x}\middle| X\right)$$ 
below and compare this with the expression for $$p\left(\widetilde{x}\middle| X\right)$$ in the cases of maximum likelihood and 
maximum a posteriori estimation. $$ \\ \begin{align*} p\left(\widetilde{x}\middle| X\right)=\int_{\theta}{p\left(\widetilde{x}\middle|\theta\right)P\left(\theta\middle| X\right)d}\theta \end{align*} \\$$
Notice that in some cases $$P\left(\theta\middle| X\right)$$ could have multiple maxima at different values of $$\theta$$ in which 
case replacing the integral with only a single point estimate of $$\theta$$ may not be a very savvy approximation to make. Alternatively, 
for some $$\widetilde{x}$$, $$p\left(\widetilde{x}\middle|\theta\right)$$ could be higher even when $$P\left(\theta\middle| X\right)$$ 
may not be high which tells us that integrating over all possible values of $$\theta$$ only can evaluate $$p\left(\widetilde{x}\middle| X\right)$$ 
accurately. The full bayesian approach uses the above integral for parameter estimation and hence is a much better predictor than 
maximum likelihood or maximum a posteriori estimators. However, it is also computationally hard to solve the integral due to the 
presence of the posterior $$P\left(\theta\middle| X\right)$$ in it. In the maximum likelihood and maximum a posteriori estimation 
we were able to get away with fully computing the denominator $$P\left(X\right)$$ in the Bayes theorem equation for 
$$P\left(\theta\middle| X\right)$$ by considering it to be a constant that would not figure in the problem of maximizing the posterior. 
However in the full bayesian approach, computing the posterior in its entirety including the denominator $$P\left(X\right)$$ 
becomes essential. And computing $$P\left(X\right)$$ is hard since $$P\left(X\right)=\int_{\theta}{P\left(X\middle|\theta\right)P\left(\theta\right)d}\theta$$. 
As we will see with the example of the coin toss experiment below, this computation may not simplify easily unless we choose the 
prior distribution $$P\left(\theta\right)$$ over the parameters $$\theta$$ that is a conjugate pair to the likelihood distribution, 
$$P\left(X\middle|\theta\right)$$ which is used to model the given data samples $$X$$. In the full Bayesian approach, the knowledge 
of the parameters is not obtained from point estimates $$\widehat{\theta_{ML}}$$ or $$\widehat{\theta_{MAP}}$$ but from the entire 
posterior probability distribution $$P\left(\theta\middle| X\right)$$ defined over the parameters.

Let us now substitute the probabilities in the Bayes' theorem $$P\left(\theta\middle| X\right)=\frac{P\left(X\middle|\theta\right).P\left(\theta\right)}{P\left(X\right)}=\frac{P\left(X\middle|\theta\right).P\left(\theta\right)}{\sum_{\theta}{P\left(X\middle|\theta\right)}.P\left(\theta\right)}$$ for the specific case of the coin toss experiment. 
The posterior is given by $$P\left(\rho\middle| C,\alpha,\beta\right)=\frac{\left(\prod_{i=1}^{N}{p\left(C=c_i\middle|\rho\right)}\right)P\left(\rho\middle|\alpha,\beta\right)}{\int_{0}^{1}\prod_{i=1}^{N}{p\left(C=c_i\middle|\rho\right)p\left(\rho\middle|\alpha,\beta\right)d}\rho}$$. 
For the specific choice of Bernoulli likelihood and beta prior distributions, this complex expression simplifies to $$\frac{\rho^{n^{\left(1\right)}}\left(1-\rho\right)^{n^{\left(0\right)}}\rho^{\alpha-1}\left(1-\rho\right)^{\beta-1}}{B\left(n^{\left(1\right)}+\alpha,n^{\left(0\right)}+\beta\right)}$$. 
Notice that this indeed computes to another beta distribution $$Beta\left(n^{\left(1\right)}+\alpha,n^{\left(0\right)}+\beta\right)$$. 
Thus we see how the Bernoulli likelihood $$\left(\prod_{i=1}^{N}{p\left(C=c_i\middle|\rho\right)}\right)$$ acts on the Beta prior 
distribution, $$Beta\left(\alpha,\beta\right)$$ to give a Beta posterior distribution $$Beta\left(n^{\left(1\right)}+\alpha,n^{\left(0\right)}+\beta\right)$$ 
which looks similar to the prior distribution that has been updated by the counts of heads $$n^{\left(1\right)}$$ and the counts of 
tails $$n^{\left(0\right)}$$ in the given data $$X$$. Choosing a conjugate pair of distributions for the likelihood and the prior 
thus enables a seamless online updation of the parameter distribution given new data samples.

### Zooming into Gaussian and Mixture of Gaussians distributions:
#### The Gaussian distribution
The Gaussian, also known as the normal distribution, is a widely used model for the distribution of continuous random variables. In the case 
of a single variable $$x$$, the Gaussian distribution can be written in the form $$ \\ \begin{align*} \mathcal{N}\left(x\middle|\mu,\sigma^2\right)=\frac{1}{\left(2\pi\sigma^2\right)^\frac{1}{2}}exp\{-\frac{1}{2\sigma^2}\left(x-\mu\right)^2\} \end{align*} \\ $$
where $$\mu$$, the mean and $$\sigma^2$$, the variance are the parameters of the univariate gaussian distribution. 
For a $$D$$-dimensional vector $$x$$, the multivariate gaussian takes the form, $$ \\ \begin{align*}\mathcal{N}\left(x\middle|\mu,\Sigma\right)=\frac{1}{\left(2\pi\right)^\frac{D}{2}}\frac{1}{\left|\Sigma\right|^\frac{1}{2}}exp\{-{\frac{1}{2}\left(x-\mu\right)}^T\Sigma^{-1}\left(x-\mu\right)\} \end{align*} \\$$
where $$\mu$$ is a $$D$$-dimensional mean vector, $$\Sigma$$ is a $$D\times D$$ covariance matrix, and $$\left|\Sigma\right|$$ denotes the 
determinant of $$\Sigma$$.

##### Maximum likelihood for the Gaussian
Given a data set $$\mathcal{X}=\left(x_1,x_2,\ldots,x_N\right)^T$$ in which the observations $$\{x_n\}$$ are assumed to be drawn independently from a multivariate Gaussian distribution, we can estimate the parameters of the distribution by maximum likelihood. The log likelihood function is given by 

