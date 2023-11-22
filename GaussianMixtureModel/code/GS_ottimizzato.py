'''
This file contain the whole implementation of a Gibbs Sampler for a Gaussian Mixture Model
If you want to see more precisely the model formulations,the updates,... look at the notebooks
'''
##### importing necessaries packages #####
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import jax
# !pip install mycolorpy
from scipy.stats import multivariate_normal
import time
from IPython.utils.sysinfo import num_cpus

##### DATA GENERATION #####
# numero di cluster
K=10

# dimensione del campione
d=2

# numero di sample
N=200

seed=2021
key = random.PRNGKey(seed)
# vectors of mean of clusters
sigma=2
mu=random.normal(key,(K,d))*sigma

seed=2023
# cluster assignment
key = random.PRNGKey(seed)
c=random.categorical(key,(1/K)*jnp.ones(shape=(K,)),axis=0,shape=(N,))
C=np.zeros(shape=(N,K))
for i in range(N):
  C[i,c[i]]=1
C=jnp.array(C)

# Data
X=np.array(np.matmul(C,mu)+random.normal(key,(N,d)))
