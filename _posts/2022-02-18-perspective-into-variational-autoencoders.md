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
several instances of the measured random variable ${\{x_i\}}_{i=1}^I$, then it might take a different value on each occasion. 
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
observed data $\{x_i\}_{i=1}^I$ by making certain assumptions. This process is referred to as learning because we learn 
about the parameters $\theta$ of the model. More generally, these are also called **parameter estimation** methods since the 
goal is to estimate parameter values $\theta$ of the probability distribution that best explain the given observations $x$ 
under an assumed form of the data distribution. A related concern of parameter estimation is that of calculating the 
probability of a new datum $x^\ast$ under the resulting model. This is known as evaluating the learned model or the 
predictive distribution. 

We will now look at three methods of parameter estimation, namely the maximum likelihood estimation, 
maximum a posteriori estimation and and the full bayesian approach. Consider a probabilistic model defined by random variable(s) $$X$$ and parameter(s) $$\theta$$. Bayes’ theorem 
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

where $$\widehat{\theta}_{ML}$$ is a single, point estimate for the parameter $\theta$ that is then used to evaluate the probability of 
a new datapoint $\widetilde{x}$ given the training data $X$ by $p\left(\widetilde{x}\middle| X\right)=p\left(\widetilde{x}\middle|\widehat{\theta_{ML}}\right)$.

Consider a simple coin-toss experiment as an example. Let $C$ denote the random variable that the coin turns head. 
An outcome, $c = 1$ for the random variable $C$ therefore denotes a head on the coin and an outcome $c = 0$ denotes a 
tail on the coin. Let $\rho$ denote the probability of getting heads on tossing the coin. Choosing a Bernoulli distribution to 
model this simple coin-toss experiment gives the likelihood below, 

$$ p\left(C=c\middle|\rho\right) = \rho^c\left(1-\rho\right)^{1-c} $$
 
$$ Log-likelihood,  l\left(\rho\right) = \sum_{i=1}^{N}logp\left(C=c_i\middle|\rho\right) $$ 

$$ = n^{\left(1\right)}log\rho+n^{\left(0\right)}log\left(1-\rho\right) $$

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
A gaussian is a bad choice for the parameter $\rho$ which only takes values in $[0,1]$. A better choice for the 
distribution of $\rho$ would be a beta distribution that spans only over the $[0, 1]$ domain. The probability distribution 
for a beta distribution is as follows: 

$$ Beta\left(\rho\middle|\alpha,\beta\right)=P\left(\rho\middle|\alpha,\beta\right)=\frac{1}{B\left(\alpha,\beta\right)}\rho^{\alpha-1}\left(1-\rho\right)^{\beta-1}$$

Here $B\left(\alpha,\beta\right)$ is the beta function defined by $\frac{\Gamma\left(\alpha+\beta\right)}{\Gamma\left(\alpha\right)\Gamma\left(\beta\right)}=\frac{1}{B\left(\alpha,\beta\right)}$. 
For eg. $Beta\left(\rho\middle|1,\ 1\right)$ reduces to a uniform distribution or a non-informative prior. An informative prior 
to encode a fair coin in our example could be denoted by say, $Beta\left(\rho\middle|5,\ 5\right)$. Note that the $\alpha$ 
and $\beta$ parameters of the beta distribution act like pseudo-counts of the outcomes, ie. the Heads and Tails of the coin toss 
experiment. 

Let us now write down the defining statement for the maximum a-posteriori estimate as $\widehat{\theta_{MAP}}=argmax_\theta p\left(\theta\middle| X\right)=argmax_\theta p\left(X\middle|\theta\right)p\left(\theta\right)$. 
Note that like in the case of maximum likelihood estimation we have avoided computing the entire posterior $p\left(\theta\middle| x\right)$ to find the 
parameter $\theta$ that maximizes the posterior. This is because the denominator $P\left(X\right)$ is a constant when maximizing 
the posterior with respect to $\theta$. After taking log and summing the quantity over all the datapoints, this is equivalent to 
$\widehat{\theta_{MAP}}=argmax_\theta\{\sum_{x\in X} log p\left(x\middle|\theta\right)+log\ p\left(\theta\right)\}$. For the example 
coin toss experiment, this would take the form of $argmax_{\theta} \left\[n^{\left(1\right)}log\rho+n^{\left(0\right)}log\left(1-\rho\right)+\left(\alpha-1\right)log\rho+\left(\beta-1\right)log\left(1-\rho\right)\ +\ CONST.\right\]$
Like in the maximum likelihood case, taking the derivative of this objective and equating to zero gives $\widehat{\rho_{MAP}}=\frac{n^{\left(1\right)}+\alpha-1}{n^{\left(1\right)}+n^{\left(0\right)}+\alpha+\beta-2}$. 
$${\widehat{\theta}}_{MAP}$$ is also a single, point estimate for the parameter $\theta$ that is then used to evaluate the probability 
of a new datapoint $\widetilde{x}$ given the training data $X$ by $p\left(\widetilde{x}\middle| X\right)=p\left(\widetilde{x}\middle|\widehat{\theta_{MAP}}\right)$.

##### 3. Full Bayesian: 
With maximum likelihood and maximum a posteriori estimation, we found point estimates of the parameter, 
$$\widehat{\theta_{ML}}$$ and $$\widehat{\theta_{MAP}}$$ respectively, that give a single best parameter setting that best explains 
the data that is given. However, in terms of finding the best prediction for a new data point ie. $p\left(\widetilde{x}\middle| X\right)$, 
using point estimates for $\theta$ are not very accurate. Consider the defining equation for $p\left(\widetilde{x}\middle| X\right)$ 
below and compare this with the expression for $p\left(\widetilde{x}\middle| X\right)$ in the cases of maximum likelihood and 
maximum a posteriori estimation. 

$$ p\left(\widetilde{x}\middle| X\right)=\int_{\theta}{p\left(\widetilde{x}\middle|\theta\right)P\left(\theta\middle| X\right)d}\theta $$

Notice that in some cases $P\left(\theta\middle| X\right)$ could have multiple maxima at different values of $\theta$ in which 
case replacing the integral with only a single point estimate of $\theta$ may not be a very savvy approximation to make. Alternatively, 
for some $\widetilde{x}$, $p\left(\widetilde{x}\middle|\theta\right)$ could be higher even when $P\left(\theta\middle| X\right)$ 
may not be high which tells us that integrating over all possible values of $\theta$ only can evaluate $p\left(\widetilde{x}\middle| X\right)$ 
accurately. The full bayesian approach uses the above integral for parameter estimation and hence is a much better predictor than 
maximum likelihood or maximum a posteriori estimators. However, it is also computationally hard to solve the integral due to the 
presence of the posterior $P\left(\theta\middle| X\right)$ in it. In the maximum likelihood and maximum a posteriori estimation 
we were able to get away with fully computing the denominator $P\left(X\right)$ in the Bayes theorem equation for 
$P\left(\theta\middle| X\right)$ by considering it to be a constant that would not figure in the problem of maximizing the posterior. 
However in the full bayesian approach, computing the posterior in its entirety including the denominator $P\left(X\right)$ 
becomes essential. And computing $P\left(X\right)$ is hard since $P\left(X\right)=\int_{\theta}{P\left(X\middle|\theta\right)P\left(\theta\right)d}\theta$. 
As we will see with the example of the coin toss experiment below, this computation may not simplify easily unless we choose the 
prior distribution $P\left(\theta\right)$ over the parameters $\theta$ that is a conjugate pair to the likelihood distribution, 
$P\left(X\middle|\theta\right)$ which is used to model the given data samples $X$. In the full Bayesian approach, the knowledge 
of the parameters is not obtained from point estimates $$\widehat{\theta_{ML}}$$ or $$\widehat{\theta_{MAP}}$$ but from the entire 
posterior probability distribution $P\left(\theta\middle| X\right)$ defined over the parameters.

Let us now substitute the probabilities in the Bayes' theorem $P\left(\theta\middle| X\right)=\frac{P\left(X\middle|\theta\right).P\left(\theta\right)}{P\left(X\right)}=\frac{P\left(X\middle|\theta\right).P\left(\theta\right)}{\sum_{\theta}{P\left(X\middle|\theta\right)}.P\left(\theta\right)}$ for the specific case of the coin toss experiment. 
The posterior is given by $P\left(\rho\middle| C,\alpha,\beta\right)=\frac{\left(\prod_{i=1}^{N}{p\left(C=c_i\middle|\rho\right)}\right)P\left(\rho\middle|\alpha,\beta\right)}{\int_{0}^{1}\prod_{i=1}^{N}{p\left(C=c_i\middle|\rho\right)p\left(\rho\middle|\alpha,\beta\right)d}\rho}$. 
For the specific choice of Bernoulli likelihood and beta prior distributions, this complex expression simplifies to $\frac{\rho^{n^{\left(1\right)}}\left(1-\rho\right)^{n^{\left(0\right)}}\rho^{\alpha-1}\left(1-\rho\right)^{\beta-1}}{B\left(n^{\left(1\right)}+\alpha,n^{\left(0\right)}+\beta\right)}$. 
Notice that this indeed computes to another beta distribution $Beta\left(n^{\left(1\right)}+\alpha,n^{\left(0\right)}+\beta\right)$. 
Thus we see how the Bernoulli likelihood $\left(\prod_{i=1}^{N}{p\left(C=c_i\middle|\rho\right)}\right)$ acts on the Beta prior 
distribution, $Beta\left(\alpha,\beta\right)$ to give a Beta posterior distribution $Beta\left(n^{\left(1\right)}+\alpha,n^{\left(0\right)}+\beta\right)$ 
which looks similar to the prior distribution that has been updated by the counts of heads $n^{\left(1\right)}$ and the counts of 
tails $n^{\left(0\right)}$ in the given data $X$. Choosing a conjugate pair of distributions for the likelihood and the prior 
thus enables a seamless online updation of the parameter distribution given new data samples.

### Zooming into Gaussian and Mixture of Gaussians distributions:
#### The Gaussian distribution
The Gaussian, also known as the normal distribution, is a widely used model for the distribution of continuous random variables. In the case 
of a single variable $x$, the Gaussian distribution can be written in the form 

$$ \mathcal{N}\left(x\middle|\mu,\sigma^2\right)=\frac{1}{\left(2\pi\sigma^2\right)^\frac{1}{2}}exp\{-\frac{1}{2\sigma^2}\left(x-\mu\right)^2\} $$

where $\mu$, the mean and $\sigma^2$, the variance are the parameters of the univariate gaussian distribution. 
For a $D$-dimensional vector $x$, the multivariate gaussian takes the form, 

$$ \mathcal{N}\left(x\middle|\mu,\Sigma\right)=\frac{1}{\left(2\pi\right)^\frac{D}{2}}\frac{1}{\left|\Sigma\right|^\frac{1}{2}}exp\{-{\frac{1}{2}\left(x-\mu\right)}^T\Sigma^{-1}\left(x-\mu\right)\} $$

where $\mu$ is a $D$-dimensional mean vector, $\Sigma$ is a $D\times D$ covariance matrix, and $\left|\Sigma\right|$ denotes the 
determinant of $\Sigma$.

##### Maximum likelihood for the Gaussian
Given a data set $\mathcal{X}=\left(x_1,x_2,\ldots,x_N\right)^T$ in which the observations $\{x_n\}$ are assumed to be drawn independently from a 
multivariate Gaussian distribution, we can estimate the parameters of the distribution by maximum likelihood. The log likelihood function is given by 

$$ ln\ p\left(X\middle|\mu,\Sigma\right)=-\frac{ND}{2}ln\left(2\pi\right)-\frac{N}{2}ln\left|\Sigma\right|-\frac{1}{2}\sum_{n=1}^{N}{\left(x_n-\mu\right)^T\Sigma^{-1}\left(x_n-\mu\right)} $$

The derivative of the log likelihood with respect to $\mu$ is given by 

$$ \frac{\partial}{\partial\mu}ln\ p\left(X\middle|\mu,\Sigma\right)=\sum_{n=1}^{N}{\Sigma^{-1}\left(x_n-\mu\right)} $$

Setting this derivative to zero, we obtain the solution for the maximum likelihood estimate of the mean given by 

$$ \mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_n $$

which is the mean of the observed data points. The maximization of the log likelihood with respect to $\Sigma$ is rather more involved since there is a symmetry constraint on the covariance 
matrix $\Sigma$. The simplest approach is to ignore the constraint and show that the resulting solution is symmetric as required. The result is as expected and takes the form

$$ \Sigma_{ML}=\frac{1}{N}\sum_{n=1}^{N}{\left(x_n-\mu_{ML}\right)\left(x_n-\mu_{ML}\right)^T} $$

This expression involves $\mu_{ML}$ because it is the result of a joint maximization with respect to $\mu$ and 
$\Sigma$. Note that the solution for $\mu_{ML}$ does not depend on $\Sigma_{ML}$, and so we can first evaluate 
$\mu_{ML}$ and then use this to evaluate $\Sigma_{ML}$.

##### Mixtures of Gaussians

While the Gaussian distribution has some important analytical properties, it suffers from significant limitations 
when it comes to modelling real data sets that are invariably multimodal having multiple dominant clumps in the 
data distribution. A simple Gaussian is unimodal and is unable to capture this structure, whereas a linear superposition 
of two or more Gaussians can characterize such a data set better. Such superpositions, formed by taking linear 
combinations of more basic distributions such as Gaussians, can be formulated as probabilistic models known as 
mixture distributions. We can visualize how powerful such mixture distributions can be in modelling real data. 
By using a sufficient number of Gaussians, and by adjusting their means and covariances as well as the coefficients 
in the linear combination, almost any continuous density can be approximated to arbitrary accuracy. We therefore 
consider a superposition of $K$ Gaussian densities of the form

$$ p\left(x\right)=\sum_{k=1}^{K}{\pi_k\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)} $$

which is called a mixture of Gaussians or a Gaussian Mixture Model (GMM). Each Gaussian density 
$\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)$ is called a component of the mixture and has its own mean $\mu_k$ 
and covariance $\Sigma_k$. The parameters $\pi_k$ are called the mixing coefficients. If we integrate both the sides 
of the above equation with respect to $x$, and note that both $p\left(x\right)$ and the individual Gaussian components 
are normalized, we obtain

$$ \sum_{k=1}^{K}\pi_k=1 $$

Also, the requirement that $p\left(x\right)\geq 0$, together with $\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)\geq0$, implies $\pi_k\geq0$ for all $k$. 
Combining the two conditions on $\pi_k$ gives $0\le\pi_k\le1$. We therefore see that the mixing coefficients satisfy the requirements to be probabilities.

The form of the Gaussian mixture distribution is governed by the parameters $\pi$, $\mu$ and $\Sigma$, where we have 
used the notation $\pi\equiv\{\pi_1,\ldots,\pi_K\}$, $\mu\equiv\{\mu_1,\ldots,\mu_K\}$ and $\Sigma\equiv\{\Sigma_1,\ldots,\Sigma_K\}$ 
with $K$ being the number of mixture components. One way to set the values of these parameters is to use maximum likelihood. The log of the likelihood is given by

$$ ln\ p\left(X\middle|\pi,\mu,\Sigma\right)=\sum_{n=1}^{N}ln\{\sum_{k=1}^{K}{\pi_k\mathcal{N}}\left(x_n\middle|\mu_k,\Sigma_k\right)\} $$

where $X=\{x_1,\ldots,x_N\}$. Notice that maximizing this expression is more complex than with a single Gaussian due to 
the presence of the summation over $k$ inside the logarithm. As a result, the maximum likelihood solution for the 
parameters no longer has a closed-form analytical solution.

##### Maximum likelihood for mixtures of Gaussians

We have seen that the Gaussian mixture distribution can be written as a linear superposition of Gaussians in the form

$$ p\left(x\right)=\sum_{k=1}^{K}{\pi_k\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)} $$

Writing the expression for the log of the likelihood for a mixture of Gaussians

$$ ln\ p\left(X\middle|\pi,\mu,\Sigma\right)=\sum_{n=1}^{N}ln\{\sum_{k=1}^{K}{\pi_k\mathcal{N}\left(x_n\middle|\mu_k,\Sigma_k\right)\}} $$

Setting the derivatives of $ln\ p\left(X\middle|\pi,\mu,\Sigma\right)$ with respect to the means $\mu_k$ of the Gaussian components to zero, 
we obtain

$$ 0=-\sum_{n=1}^{N}\frac{\pi_k\mathcal{N}\left(x_n\middle|\mu_k,\Sigma_k\right)}{\sum_{j}{\pi_j\mathcal{N}\left(x_n\middle|\mu_j,\Sigma_j\right)}}\Sigma_k\left(x_n-\mu_k\right) $$

$$ =-\sum_{n=1}^{N}{\gamma\left(z_{nk}\right)\Sigma_k\left(x_n-\mu_k\right)} $$

The $\gamma\left(z_{nk}\right)$ are called responsibilities and are in fact posterior probabilities $p\left(z_{nk}=1\middle| x\right)$ 
that can be found using the Bayes’ theorem

$$ \gamma\left(z_{nk}\right)\equiv p\left(z_{nk}=1\middle| x_n\right) $$

$$ =\frac{p\left(z_{nk}=1\right)p\left(x_n\middle| z_{nk}=1\right)}{\sum_{j=1}^{K}{p\left(z_{nj}=1\right)p\left(x_n\middle| z_{nj}=1\right)}} $$

$$ =\frac{\pi_k\mathcal{N}\left(x_n\middle|\mu_k,\Sigma_k\right)}{\sum_{j=1}^{K}{\pi_j\mathcal{N}\left(x_n\middle|\mu_j,\Sigma_j\right)}} $$

Coming back to the maximization, we have

$$ 0=-\sum_{n=1}^{N}{\gamma\left(z_{nk}\right)\Sigma_k\left(x_n-\mu_k\right)} $$

Multiplying by $\Sigma_k^{-1}$ (which we assume to be non-singular) and rearranging we obtain

$$ \mu_k=\frac{1}{N_k}\sum_{n=1}^{N}{\gamma\left(z_{nk}\right)x_n} $$

where we have defined $\ N_k=\sum_{n=1}^{N}\gamma\left(z_{nk}\right)$, $N_k$ can be interpreted as the effective number of data points assigned to cluster $k$.
Now let us try to set the derivative of $ln\ p\left(X\middle|\pi,\mu,\Sigma\right)$ with respect to $\Sigma_k$ to zero, 
and follow a similar line of reasoning, making use of the result for the maximum likelihood solution for the covariance matrix of a single Gaussian, we obtain

$$ \Sigma_k=\frac{1}{N_k}\sum_{n=1}^{N}{\gamma\left(z_{nk}\right)\left(x_n-\mu_k\right)\left(x_n-\mu_k\right)^T} $$

Finally, we maximize $ln\ p\left(X\middle|\pi,\mu,\Sigma\right)$ with respect to the mixing coefficients $\pi_k$. 
Here we must take account of the constraint $\sum_{k=1}^{K}\pi_k=1$ which requires the mixing coefficients to sum to one. 
This can be achieved using a Lagrange multiplier and by maximizing the following quantity

$$ ln\ p\left(X\middle|\pi,\mu,\Sigma\right)+\lambda\left(\sum_{k=1}^{K}\pi_k-1\right) $$

which gives

$$ 0=\sum_{n=1}^{N}\frac{\mathcal{N}\left(x_n\middle|\mu_k,\Sigma_k\right)}{\sum_{j}{\pi_j\mathcal{N}\left(x_n\middle|\mu_j,\Sigma_j\right)}}+\ \lambda $$

Where again we see the appearance of the responsibilities. If we now multiply both sides by $\pi_k$ and sum over $k$ making use of the constraint 
$\sum_{k=1}^{K}\pi_k=1$, we find $\lambda=\ -N$. Using this to eliminate $\lambda$ and rearranging we obtain the 
following expression for the mixing coefficients

$$ \pi_k=\frac{N_k}{N} $$

Thus we have obtained expressions for the parameters $\pi_k$, $\mu_k$ and $\Sigma_k$ for all $k\ =\ 1\ldots K$. 
It is worth emphasizing that the result expressions do not constitute a closed form solution for the parameters of the mixture 
model because the responsibilities $\gamma\left(z_{nk}\right)$ depend on those parameters in a complex way by its definition equation 
which we saw earlier in this section. However, these results do suggest a simple iterative scheme for finding a solution to the maximum 
likelihood problem, which as we shall see turns out to be an instance of the Expectation Maximization(EM) algorithm for the particular 
case of the Gaussian mixture model. We first choose some initial values for the means, covariances and mixing 
coefficients. Then we alternate between the following two updates as shown in the algorithm below

##### EM for Gaussian mixtures

Given a Gaussian mixture model, the goal is to maximize the likelihood function with respect to the parameters (comprising the means and covariances of the components and the mixing coefficients). 

Step 1. Initialize the means $\mu_k$, covariances $\Sigma_k$ and mixing coefficients $\pi_k$, and evaluate the initial value of the log likelihood.
Step 2. The E step. Evaluate the responsibilities using the current parameter values.

$$ \gamma\left(z_{nk}\right)=\frac{\pi_k\mathcal{N}\left(x_n\middle|\mu_k,\Sigma_k\right)}{\sum_{j=1}^{K}{\pi_j\mathcal{N}\left(x_n\middle|\mu_j,\Sigma_j\right)}}$$

Step 3. The M step. Re-estimate the parameters using the current responsibilities.

$$ \mu_k^{new}=\frac{1}{N_k}\sum_{n=1}^{N}{\gamma\left(z_{nk}\right)x_n} $$

$$ \Sigma_k^{new}=\frac{1}{N_k}\sum_{n=1}^{N}{\gamma\left(z_{nk}\right)\left(x_n-\mu_k^{new}\right)\left(x_n-\mu_k^{new}\right)^T}$$

$$ \pi_k^{new}=\frac{N_k}{N} $$

where

$$ N_k=\sum_{n=1}^{N}\gamma\left(z_{nk}\right) $$

Step 4. Evaluate the log likelihood

$$ ln\ p\left(X\middle|\mu,\Sigma,\pi\right)=\sum_{n=1}^{N}ln\{\sum_{k=1}^{K}{\pi_k\mathcal{N}\left(x_n\middle|\mu_k,\Sigma_k\right)\}} $$

and check for convergence of either the parameters or the log likelihood. If the convergence criterion is not satisfied return to step 2.

In the later sections we shall show that each update to the parameters resulting from an E step followed by an M step is guaranteed 
to increase the log likelihood function. In practice, the algorithm is deemed to have converged when the change in the log likelihood 
function, or alternatively in the parameters, falls below some threshold.

Till now, we motivated the Gaussian mixture model as a simple linear superposition of Gaussian components aimed 
at providing a richer class of density models than the single Gaussian. We now turn to a latent variable view of 
mixture distributions in which the discrete latent variables can be interpreted as defining assignments of data 
points to specific components of the mixture. This will provide us with a deeper insight into this important 
distribution and will also serve to motivate the expectation-maximization algorithm. Note in this context that a 
general technique for finding maximum likelihood estimators in latent variable models is the expectation-maximization(EM) 
algorithm.

Let us now see how Gaussian mixtures can be interpreted in terms of latent random variables. For that, let us first 
understand the process of generating a sample from a gaussian mixture distribution and try to emulate that through 
a probabilistic graphical model. A typical procedure to sample from a Gaussian mixture would be to first pick a mixture 
component $k$ from the $K$ mixture components with probability $\pi_k$ following which a Gaussian associated with the $k\ th$ 
component having mean $\mu_k$ and covariance $\Sigma_k$ is sampled to generate an observation. To give an example in the 
context of the coin-toss experiment, consider the case where there are multiple coins each one having its own unique 
probability of showing a head. The data given to us is just a sequence of observations of Heads or Tails that turned up 
in the experiment without information about which coin was tossed to give that observation. Here, the identity of the coin 
becomes a hidden or latent random variable and the observation of the Heads or Tails that is given in the data becomes the 
observed random variable.

Consider the graphical representation below of this mixture model involving two random variables, $X$ and $Z$. 
$X$ is a continuous random variable and $Z$ is a discrete random variable.

Now consider the graphical representation of the specific case of the Gaussian mixture model. Here $\pi$, $\mu$ and $\Sigma$ 
are the parameters of the model, and $N$ in the bottom right corner of the square box denotes that this dependency relation 
between $Z$ and $X$ holds independently for all the $N$ points in the data set.

In particular, $Z$ for each data point is a $K$- dimensional binary random variable having a $1-of-K$ representation in 
which a particular element $z_k$ is equal to $1$ and all other elements are equal to $0$. The values of $z_k$ therefore 
satisfy $z_k\ \in\{0,\ 1\}$ and $\sum_{k} z_k=1$, and we see that there are $K$ possible states for the vector $z$ according 
to which element is nonzero. We are also given that $p\left(z_k=1\right)=\pi_k$. This means that we could now model $Z$ through 
a categorical  distribution. $X$ is a continuous random variable generated by sampling from a normal distribution 
$\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)$ depending on the mixture component $k$ that was picked. Looking at the entire 
graphical model as a single, blackbox probability distribution that generates samples $x$, we have considered $X$ as the continuous 
random variable that is observed and $Z$ as the discrete random variable that is latent or ‘hidden’ in the observations. In the case 
of a Gaussian mixture model, the latent variable denotes the mixture component that was chosen to generate the observation $x$.

The distribution of $z$ can be written as 

$$ p\left(z\right)=\prod_{k=1}^{K}\pi_k^{z_k} $$

where $z=\left(z_1,z_2,\ldots,z_K\right)$ is a categorical random variable with success probabilities defined by $p\left(z_k=1\right)=\pi_k$ 
such that the parameters $\pi_k$ must satisfy $0\le\pi_k\le1$ and $\sum_{k=1}^{K}\pi_k=1$ to be valid probabilities.

Similarly, the conditional distribution of $x$ given a particular value for $z$ is a Gaussian

$$ p\left(x\middle| z_k=1\right)=\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right) $$

This can also be written in the form

$$ p\left(x\middle| z\right)=\prod_{k=1}^{K}{\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)}^{z_k} $$

The joint distribution is given by $p\left(x,z\right)=p\left(z\right)p\left(x\middle| z\right)$, and the marginal distribution of $x$ 
is then obtained by summing the joint distribution over all possible states of $z$ to give

$$ p\left(x\right)=\sum_{z}{p\left(z\right)p\left(x\middle| z\right)}=\sum_{k=1}^{K}{\pi_k\mathcal{N}\left(x\middle|\mu_k,\Sigma_k\right)} $$

Thus, the marginal distribution of $x$ is a Gaussian mixture! If we have several observations $x_1,x_2,\ldots,x_N$, then, because we have 
represented the marginal distribution in the form $p\left(x,z\right)=\sum_{z} p\left(x,z\right)$, it follows that for every observed data point 
$x_n$, there is a corresponding latent variable $z_n$.

We have therefore found an equivalent formulation of the Gaussian mixture involving an explicit latent variable $z$. 
Following in this paragraph are some points to ponder about the role of latent variables from our discussion above. 
The primary role of the latent variables is to allow a complicated distribution over the observed variables to be 
represented in terms of a model constructed from simpler conditional distributions. If we define a joint distribution 
over observed and latent variables, the corresponding distribution of the observed variables alone is obtained by 
marginalization. This allows relatively complex marginal distributions over observed variables to be expressed in 
terms of more tractable joint distributions over the expanded space of observed and latent variables. The introduction 
of latent variables thereby allows complicated distributions to be formed from simpler components.

### The EM algorithm in General

In the previous section, we discussed in detail the steps of the EM algorithm to find the maximum likelihood solution for 
the specific case of Gaussian mixture models. We also followed that up with illustrating how a Gaussian mixture model can 
be interpreted in terms of a probabilistic graphical model with an explicit latent variable. In this section, we will try 
to provide an intuitive (and beautiful!) proof of the EM algorithm in general for any general probabilistic model with latent 
variables. Consider such a model in which we collectively denote all of the observed variables by $X$ and all of the latent or 
hidden variables by $Z$. The joint distribution $p\left(X,Z\middle|\theta\right)$ is governed by a set of parameters denoted $\theta$. 
Our goal is to maximize the likelihood function that is given by

$$ p\left(X\middle|\theta\right)=\sum_{Z}{p\left(X,Z\middle|\theta\right)} $$

Here we are assuming $Z$ is discrete, although the discussion is identical if $Z$ comprises continuous variables or a combination of 
discrete and continuous variables, with summation replaced by integration as appropriate. We shall suppose that direct optimization of 
$p\left(X\middle|\theta\right)$ is difficult, but that optimization of the complete-data log likelihood function $p\left(X,Z\middle|\theta\right)$ 
is significantly easier. 

Next, we introduce a distribution $q\left(Z\right)$ defined over the latent variables, and we observe that for any choice of $q\left(Z\right)$, 
the following decomposition relation holds,

$$ ln\ p(X|\theta)\ =\ \mathcal{L}(q,\ \theta)\ +\ KL(q||p) $$

where we have defined

$$ \mathcal{L}\left(q,\theta\right)=\sum_{Z} q\left(Z\right)ln\{\frac{p\left(X,Z\middle|\theta\right)}{q\left(Z\right)}\} $$

$$ KL(q||p)\ =\ -\ \sum_{Z}{q\left(Z\right)ln\ \{\frac{p\left(Z\middle| X,\theta\right)}{q\left(Z\right)}\}} $$

Note that $\mathcal{L}\left(q,\theta\right)$ is a functional  of the distribution $q\left(Z\right)$, and a function of the 
parameters $\theta$. Also, to verify that the above decomposition holds, we first make use of the product rule of probability 
to give

$$ ln\ p\left(X,Z\middle|\theta\right)=ln\ p\left(Z\middle| X,\theta\right)+ln\ p\left(X\middle|\theta\right) $$

which we then substitute into the expression for $\mathcal{L}\left(q,\theta\right)$. This gives rise to two terms, one of which 
cancels $KL(q||p)$ while the other gives the required log likelihood $ln\ p\left(X\middle|\theta\right)$ after noting that 
$q\left(Z\right)$ is a normalized distribution that sums to $1$. Also note that $KL(q||p)$ is the Kullback-Leibler divergence 
between $q\left(Z\right)$ and the posterior distribution $p\left(Z\middle| X,\theta\right)$. Recall that the Kullback-Leibler 
divergence satisfies $KL(q||p)\ \geq0$, with equality if, and only if, $q\left(Z\right)=p\left(Z\middle| X,\theta\right)$. It 
therefore follows from the decomposition relation for $ln\ p\left(X\middle|\theta\right)\$ that $\mathcal{L}\left(q,\theta\right)\le ln\ p\left(X\middle|\theta\right)$, 
in other words that $\mathcal{L}\left(q,\theta\right)$ is a lower bound on $lnp\left(X\middle|\theta\right)$.

The EM algorithm is a two-stage iterative optimization technique for finding maximum likelihood solutions. We can use the 
decomposition relation for $ln\ p\left(X\middle|\theta\right)$ discussed above to define the EM algorithm and to demonstrate 
that it does indeed maximize the log likelihood.

Suppose that the current value of the parameter vector is $\theta^{old}$. In the E step, the lower bound 
$\mathcal{L}\left(q,\theta^{old}\right)$ is maximized with respect to $q\left(Z\right)$ while holding $\theta^{old}$ 
fixed. The solution to this maximization problem is easily seen by noting that the value of $ln\ p\left(X\middle|\theta^{old}\right)$ 
does not depend on $q\left(Z\right)$ and so the largest value of $\mathcal{L}\left(q,\theta^{old}\right)$ will occur when the 
Kullback-Leibler divergence vanishes, in other words when $q\left(Z\right)$ is equal to the posterior distribution 
$p\left(Z\middle| X,\theta^{old}\right)$. In this case, the lower bound will equal the log likelihood, as illustrated in the figure below. 

In the subsequent M step, the distribution $q\left(Z\right)$ is held fixed and the lower bound $\mathcal{L}\left(q,\theta\right)$ is 
maximized with respect to $\theta$ to give some new value $\theta^{new}$. This will cause the lower bound $\mathcal{L}$ to increase 
(unless it is already at a maximum), which will necessarily cause the corresponding log likelihood function to increase. Because the 
distribution $q$ is determined using the old parameter values rather than the new values and is held fixed during the M step, it will 
not equal the new posterior distribution $p\left(Z\middle| X,\theta^{new}\right)$, and hence there will be a nonzero KL divergence. 
The increase in the log likelihood function is therefore greater than the increase in the lower bound, as shown in figure below.

If we substitute $q\left(Z\right)=p\left(Z\middle| X,\theta^{old}\right)$ into the expression for $\mathcal{L}\left(q,\theta\right)$ 
defined after the decomposition relation listed earlier, we see that after the E step, the lower bound takes the form

$$ \mathcal{L}\left(q,\theta\right)=\sum_{Z}{p\left(Z\middle| X,\theta^{old}\right)ln}p\left(X,Z\middle|\theta\right)-\sum_{Z}{p\left(Z\middle| X,\theta^{old}\right)ln}p\left(Z\middle| X,\theta^{old}\right) $$

$$ =\mathcal{Q}\left(\theta,\theta^{old}\right)\ +\ const $$

where the constant is simply the negative entropy of the $q$ distribution and is therefore independent of $\theta$. 
Thus, in the M step, the quantity that is being maximized is the expectation of the complete-data log likelihood, as 
we saw earlier in the specific case of mixture of Gaussians. Note that the variable $\theta$ over which we are optimizing 
appears only inside the logarithm. If the joint distribution $p\left(Z\middle| X,\theta\right)$ comprises a member of the 
exponential family, or a product of such members, then we see that the logarithm will cancel the exponential and lead to 
an M step that will typically be much simpler than the maximization of the corresponding incomplete-data log likelihood function 
$p\left(X\middle|\theta\right)$.

The operation of the EM algorithm can also be viewed in the parameter space, as illustrated in the figure below.

Here the red curve depicts the (incomplete data) log likelihood function whose value we wish to maximize. We start with some 
initial parameter value $\theta^{old}$, and in the first E step we evaluate the posterior distribution over latent variables, 
which gives rise to a lower bound $\mathcal{L}\left({\theta,\ \theta}^{old}\right)$ whose value equals the log likelihood at 
$\theta^{old}$, as shown by the blue curve. Note that the bound makes a tangential contact with the log likelihood at 
$\theta^{old}$, so that both curves have the same gradient. This bound is a convex function having a unique maximum (for mixture 
components from the exponential family). In the M step, the bound is maximized giving the value $\theta^{new}$, which gives a 
larger value of log likelihood than $\theta^{old}$. The subsequent E step then constructs a bound that is tangential at $\theta^{new}$ 
as shown by the green curve.

Thus, we have seen that both the E and the M steps of the EM algorithm are increasing the value of a well-defined bound on the log 
likelihood function and that the complete EM cycle will change the model parameters in such a way as to cause the log likelihood to 
increase (unless it is already at a maximum, in which case the parameters remain unchanged). We can also use the EM algorithm to maximize 
the posterior distribution $p\left(\theta\middle| X\right)$ for models in which we have introduced a prior $p\left(\theta\right)$ over the 
parameters. To see this, we note that as a function of $\theta$, we have $p\left(\theta\middle| X\right)=p\left(\theta,X\right)/p\left(X\right)$ 
and so

$$ ln\ p\left(\theta\middle| X\right)=ln\ p\left(\theta,X\right)-ln\ p\left(X\right) $$

Making use of the decomposition relation of $ln\ p\left(X\middle|\theta\right)$ which we saw earlier, we have

$$ ln\ p(\theta|X)\ =\ \mathcal{L}(q,\ \theta)\ +\ KL(q||p)\ +\ ln\ p(\theta)\ -\ ln\ p(X) $$

$$ \geq\mathcal{L}\left(q,\theta\right)+ln\ p\left(\theta\right)-ln\ p\left(X\right) $$

where $ln\ p\left(X\right)$ is a constant. We can again optimize the right-hand side alternately with respect to $q$ and 
$\theta$. The optimization with respect to $q$ gives rise to the same E-step equations as for the standard EM algorithm, 
because $q$ only appears in $\mathcal{L}\left(q,\theta\right)$. The M-step equations are modified through the introduction 
of the prior term $ln\ p\left(\theta\right)$, which typically requires only a small modification to the standard maximum 
likelihood M-step equations.

The EM algorithm therefore breaks down the potentially difficult problem of maximizing the likelihood function into two 
stages, the E step and the M step, each of which will often prove simpler to implement. Nevertheless, for complex models 
it may be the case that either the E step or the M step, or indeed both, remain intractable! Let us now list the steps of 
the EM algorithm for a general probabilistic graphical model.

##### The General EM algorithm

Given a joint distribution $p\left(X,Z\middle|\theta\right)$ over observed variables $X$ and latent variables $Z$, 
governed by parameters $\theta$, the goal is to maximize the likelihood function $p\left(X\middle|\theta\right)$ with 
respect to $\theta$.

Step 1. Choose an initial setting for the parameters $\theta^{old}$.
Step 2. E step: Evaluate $p\left(Z\middle| X,\theta^{old}\right)$
Step 3. M step: Evaluate $\theta^{new}$ given by

$$ \theta^{new}=argmax_\theta\ \mathcal{Q}\left(\theta,\theta^{old}\right) $$

where 

$$ \mathcal{Q}\left(\theta,\theta^{old}\right)=\sum_{Z}{p\left(Z\middle| X,\theta^{old}\right)ln}p\left(X,Z\middle|\theta\right) $$

Step 4. Check for convergence of either the log likelihood or the parameter values. If the convergence criterion is not satisfied, 
then let $\theta^{old}\gets\theta^{new}$ and return to step 2.

Here we have considered the use of the EM algorithm to maximize a likelihood function when there are discrete 
latent variables. However, it can also be applied when the unobserved variables correspond to missing values in 
the data set and used for missing data imputation in such incomplete datasets.

At this point, it should be emphasized that there will generally be multiple local maxima of the log likelihood 
function and that EM will find one of them and not necessarily guaranteed to find the largest of these maxima. 
Also, it is worth emphasizing that there is a significant problem associated with the maximum likelihood framework 
applied to Gaussian mixture models, due to the presence of singularities. For simplicity, consider a Gaussian 
mixture whose components have covariance matrices given by $\Sigma_k={\sigma_k}^2I$, where $I$ is the unit matrix, 
although the conclusions will hold for general (non-diagonal or non-identity matrices) covariance matrices. Suppose 
that one of the components of the mixture model, let us say the $jth$ component, has its mean $\mu_j$ exactly equal 
to one of the data points so that $\mu_j=x_n$ for some value of $n$. This data point will then contribute a term in 
the likelihood function of the form  $\mathcal{N}\left(x_n\middle| x_n,{\sigma_j}^2I\right)=\frac{1}{\left(2\pi\right)^{1/2}}\frac{1}{\sigma_j}$. 
If we consider the limit $\sigma_j\rightarrow0$, then we see that this term goes to infinity and so the log likelihood 
function is not a well posed problem because such singularities will always be present and will occur whenever one of the 
Gaussian components ‘collapses’ onto a specific data point. Recall that this problem did not arise in the case of a single 
Gaussian distribution. To understand the difference, note that if a single Gaussian collapses onto a data point it will 
contribute multiplicative factors to the likelihood function arising from the other data points and these factors will go 
to zero exponentially fast, giving an overall likelihood that goes to zero rather than infinity. However once we have at 
least two components in the mixture, one of the components can have a finite variance and therefore assign finite probability 
to all of the data points while the other component can shrink onto one specific data point and thereby contribute an ever 
increasing additive value to the log likelihood as illustrated in the figure below.

These singularities provide an example of the severe over-fitting that can occur in a maximum likelihood approach. We shall see 
that this difficulty does not occur if we adopt a full Bayesian approach. In the next section, we will see how EM can be generalized 
to obtain the more elegant variational inference framework that computes a full Bayesian solution to the parameter estimation problem.

To summarize the document till this point, this is what we have been doing till now in order. We
1.	looked at the need for modelling data with probability distributions in machine learning
2.	got to know that there are different types of probability distributions available to model discrete/continuous valued data as random variables 
3.	looked at the 3 techniques for parameter estimation to fit the given data to those probability distributions ie. estimate the parameters of the probability distribution to be able to explain (or model) the given data. The 3 techniques were maximum likelihood, maximum a posteriori and full Bayesian estimation 
4.	looked at the Gaussian and motivated the need for mixture models like the mixture of Gaussians to fit complex real-world data 
5.	applied the maximum likelihood parameter estimation method to the mixture of gaussians and found that a closed form analytical solution was difficult to arrive at, and proposed the EM algorithm as a technique that can be used to find maximum likelihood solutions to mixture models
6.	interpreted mixture models in terms of the more general framework of probabilistic graphical models with latent variables. 
7.	derived the EM algorithm for a general latent variable graphical model and written down the EM algorithm in general for any latent variable model.
8.	Problems with EM and the maximum likelihood solution for the parameter estimation problem and an assurance that the variational inference framework would resolve this.   
Moving forward we will be looking at the following in order: 
1.	Building concepts in variational inference (ELBO loss).
2.	Variational inference solution for a gaussian mixture.
3.  VAEs as the variational inference solver that uses neural network models for the posterior and the likelihood probability distributions 
and minimizes the ELBO loss, which is a sum of a squared error loss(at the output of the VAE's decoder network) and a 
KL divergence loss (at the output of the VAE's encoder network and a sampled Z). Note that a simple autoencoder only minimizes 
a simple squared error loss, which is equivalent to modelling the data as a single gaussian because the log likelihood of a gaussian 
distribution reduces to a simple squared difference between the data and the mean of the gaussian, which is exactly equivalent to the 
squared error loss we are trying to minimize in a neural network. Whereas a VAE models the data as a more powerful mixture of gaussian
distributions by trying to minimize the ELBO loss!! If nothing else made sense in this entire blog except this single point, then you have 
grabbed the essence of the VAEs :-)   

...to be continued