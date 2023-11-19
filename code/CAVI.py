'''
This file contain the whole implmentation of the Coordinate Ascent Algorithm for Variational Inference on a Gaussian Mixture Model
If you want to see more precisely the model formulations,the updates,... look at the notebooks
'''
##### importing necessaries packages #####

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import jax
!pip install mycolorpy
from IPython.utils.sysinfo import num_cpus

##### DATA GENERATION #####
# numero di cluster
K=5

# dimensione del campione
d=2

# numero di sample
N=1000

seed=2022
key = random.PRNGKey(seed)
# vectors of mean of clusters
sigma=5
mu=random.normal(key,(K,d))*sigma

# cluster assignment
key = random.PRNGKey(2)
c=random.categorical(key,(1/K)*jnp.ones(shape=(K,)),axis=0,shape=(N,))
C=np.zeros(shape=(N,K))
for i in range(N):
  C[i,c[i]]=1
C=jnp.array(C)

# Data
X=jnp.matmul(C,mu)+random.normal(key,(N,d))

##### AUXILIARY FUNCTIONS #####

## Computes the update of phi
#we give two different version where the second is an optimization of the first
'''
def update_phi(data,phi,m,s2):
    for i in jnp.arange(data.shape[0]):
        for k in jnp.arange(phi.shape[1]):
          phi[i,k]=jnp.exp(jnp.matmul(m[k,:],data[i,:].transpose())-(s2[k]+jnp.matmul(m[k,:],m[k,:].transpose()))/2) # non sono così sicuro della formula per mk^2
        phi[i,:]=phi[i,:]/jnp.sum(phi[i,:])
    return phi
'''

def update_phi(data, phi, m, s2):
    
    updated_phi = jnp.zeros_like(phi)
    M=jnp.matmul(m,m.T)
    log_likelihood = jnp.matmul(data,m.T) - 0.5 * jnp.matmul(jnp.ones(shape=(N,1)),(s2.T + jnp.resize(jnp.diag(M),(K,1)).T))
    updated_phi = jnp.exp(log_likelihood)

    #normalization since phi's are probabilities
    updated_phi /= jnp.sum(updated_phi,axis=1,keepdims=True)
    return updated_phi

update_phi_jit=jit(update_phi)

# Computes the update of mean and variance
def update_mean_and_variance(data,phi,sigma):
  K=phi.shape[1]
  d=data.shape[1]
  N=data.shape[0]
  updated_m=(jnp.matmul(phi.T,data)/(1/sigma**2*jnp.ones(shape=(K,d))+jnp.matmul(jnp.resize(jnp.sum(phi,axis=0),(1,K)).T,jnp.ones(shape=(1,d)))))
  updated_s2=(1/(1/sigma**2*jnp.ones(shape=(K,1))+jnp.resize(jnp.sum(phi,axis=0),(1,K)).T))
  return updated_m,updated_s2

update_mean_and_variance_jit=jit(update_mean_and_variance)


def compute_ELBO(m,s2,phi,data):
  # when computing the ELBO value, we omit constants because once we compute the improvement they would have a total of 0
  # Fn stands for the nth component of the formula (21) in the review paper
  d=m.shape[1]
  M=jnp.matmul(m,m.T)

  F1=-0.5*sigma**2 *jnp.sum( d*s2+jnp.diag(M))

  # F2= -log(K) sum over k from 1 to K => constant in every iteration

  F3=-0.5*jnp.sum(jnp.matmul(phi.T,jnp.diag(jnp.matmul(data,data.T))))+jnp.sum(jnp.diag(jnp.matmul(phi.T,jnp.matmul(data,m.T))))-0.5*d*jnp.sum(jnp.matmul(phi,s2))
  F3+=-0.5*jnp.sum(jnp.matmul(phi,jnp.diag(M)))
  # -d/2*jnp.log(2*jnp.pi)*phi[i,k] summed over i and k should be constant over time, since phi[i,:] is a probability it should sum N every time

  F4=jnp.sum(jnp.log(phi)*phi)

  F5=-d/2*jnp.sum(jnp.log(s2))

  return F1+F3+F4+F5

compute_ELBO_jit=jit(compute_ELBO)

##### VARIATIONAL INFERENCE #####
# FUNCTION FOR VARIATIONAL INFERENCE
# Notation of the paper

def single_iteration_VI(data,K,sigma,i,nMAX,tol):
    N=data.shape[0]
    d=data.shape[1]
    key = random.PRNGKey(i*seed)
    phi=random.uniform(key,minval=0,maxval=1,shape=(N,K))
    phi/=jnp.sum(phi,axis=1,keepdims=True)
    m=random.normal(key,shape=(K,d))*sigma
    s2=random.uniform(key,minval=0,maxval=10,shape=(K,1))
    improvement=1
    ELBO_old=0 # probabilmente questo andrà modificato
    ELBO_new=compute_ELBO_jit(m,s2,phi,data)
    nit=0
    while (improvement>tol and nit<nMAX) or nit<15:
      phi=update_phi_jit(data,phi,m,s2)
      m,s2=update_mean_and_variance_jit(data,phi,sigma)
      ELBO_old=ELBO_new
      ELBO_new=compute_ELBO_jit(m,s2,phi,data)
      improvement=ELBO_new-ELBO_old
      nit+=1
      #print('Iter ',nit,'\t ELBO: ',ELBO_new,'\t Improvement: ',improvement,'\n')
      #print('=================================================\n')
    return  m,s2,phi,ELBO_new,nit

single_iteration_VI_jit=jit(single_iteration_VI,static_argnames=['K','nMAX','tol'])


def VI(data,K,sigma,nMAX,n_iniz,tol):
  # creating our variables as estimation of parameters for posterior probabilities
  # jax arrays are immutable, so I don't know how to create these variables in jax and use them
  # I iniialize them randomly, since there is no a-priori starting point which is best than others
  ELBO_max=0
  for i in range(n_iniz):
    m,s2,phi,ELBO_new,nit=single_iteration_VI(data,K,sigma,i,nMAX,tol)
    if i==0:
      ELBO_max=ELBO_new
      m_max=m
      s2_max=s2
      phi_max=phi
      n_max=0
    print('Initialization number: ',i+1,'\t ELBO: ',ELBO_new,'\t N_iterations: ',nit)
    print('=================================================\n')
    if ELBO_new>ELBO_max:
      ELBO_max=ELBO_new
      m_max=m
      s2_max=s2
      phi_max=phi
      n_max=i
  print('Best initialization at ',n_max+1,' \t ELBO: ',ELBO_max,'\n\n')
  return m_max,s2_max,phi_max

VI_jit=jit(VI)

##### RUNNING COMMANDS #####
tol=10**-16
m,s2,phi=VI(X,K,sigma,10000,100,tol)
#likelihood
n_cluster = np.sum(C, axis = 0)
m_post=1/(1+1/sigma**2)/np.matmul(n_cluster.reshape((K,1)),np.ones(shape=(1,d)))*np.matmul(C.T,X)
s2_post=1/(1+1/sigma**2)/n_cluster
