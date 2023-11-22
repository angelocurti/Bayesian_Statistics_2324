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
!pip install mycolorpy
from scipy.stats import multivariate_normal
from IPython.utils.sysinfo import num_cpus

##### DATA GENERATION #####
# numero di cluster
K=10

# dimensione del campione
d=2

# numero di sample
N=2000

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
X=jnp.matmul(C,mu)+random.normal(key,(N,d))

##### GIBBS SAMPLER #####
# The whole implementation is performed in a unique function, this because we eant to folly understand and correctly implement 
# each step of the algorithm before going for optimization
n_cluster = np.sum(C, axis = 0)
m_post=1/(1+1/sigma**2)/np.matmul(n_cluster.reshape((K,1)),np.ones(shape=(1,d)))*np.matmul(C.T,X)
s2_post=1/(1+1/sigma**2)/n_cluster

def gibbs_gaussian_mixture(data, num_clusters, sigma, num_iterations):
    # Data dimensions
    N, D = data.shape

    # Initialization
    cluster_assignments = np.random.choice(num_clusters, N)
    cluster_means_samples = np.zeros((num_iterations, num_clusters, D))
    cluster_variances_samples = np.zeros((num_iterations, num_clusters))
    cluster_probabilities_samples = np.zeros((num_iterations, N, num_clusters))

    # Initialize cluster means and variances
    cluster_means = np.random.multivariate_normal(mean=np.mean(data, axis=0), cov=np.eye(D), size=num_clusters)
    cluster_variances = np.ones(num_clusters)

    # Gibbs Sampler
    for iteration in range(num_iterations):
        # Update cluster assignments
        for i in range(N):
            # Compute probabilities for each cluster
            likelihoods = [multivariate_normal.pdf(data[i,], mean=cluster_means[k,], cov=cluster_variances[k]*np.eye(D)) for k in range(num_clusters)]

            # Update cluster assignment using categorical prior
            probabilities = likelihoods / np.sum(likelihoods)
            probabilities=np.nan_to_num(probabilities,nan=0)
            probabilities=probabilities/np.sum(probabilities)
            cluster_assignments[i] = np.random.choice(num_clusters, p=probabilities)

        # Update cluster means and variances
        for k in range(num_clusters):
            cluster_size = np.sum(cluster_assignments == k)
            prior_mean = np.zeros(D)
            prior_covariance = sigma**2 * np.eye(D)

            # Compute posterior parameters
            posterior_covariance = np.linalg.inv(np.linalg.inv(prior_covariance) + (cluster_size / np.var(data[cluster_assignments == k,], axis=0)) * np.eye(D))
            posterior_mean = posterior_covariance @ (np.linalg.inv(prior_covariance) @ prior_mean + (cluster_size / np.var(data[cluster_assignments == k,], axis=0)) * np.sum(data[cluster_assignments == k,], axis=0))

            # Update cluster means
            cluster_means[k, :] = np.random.multivariate_normal(mean=posterior_mean, cov=posterior_covariance)

            # Update cluster variances
            cluster_variances[k] = np.mean(np.var(data[cluster_assignments == k,], axis=0)) / cluster_size

            # Store samples
            cluster_means_samples[iteration, k, :] = cluster_means[k, :]
            cluster_variances_samples[iteration, k] = cluster_variances[k]

        # Compute cluster probabilities for each data point
        for i in range(N):
            for k in range(num_clusters):
                likelihood = multivariate_normal.pdf(data[i,], mean=cluster_means[k, :], cov=cluster_variances[k]*np.eye(D))
                cluster_probabilities_samples[iteration, i, k] = likelihood / np.sum([multivariate_normal.pdf(data[i,], mean=cluster_means[j, :], cov=cluster_variances[k]*np.eye(D)) for j in range(num_clusters)])

    # Compute final means and variances based on collected samples
    final_cluster_means = np.mean(cluster_means_samples, axis=0)
    final_cluster_variances = np.mean(cluster_variances_samples, axis=0)

    # Compute final cluster probabilities for each data point
    final_cluster_probabilities = np.mean(cluster_probabilities_samples, axis=0)

    return final_cluster_means, final_cluster_variances, final_cluster_probabilities

##### RUNNING COMMANDS #####
n_iterations=100
result_means, result_variances, result_probs = gibbs_gaussian_mixture(X, K, sigma, n_iterations)

print("Final Cluster Means:")
print(result_means)

print("\nFinal Cluster Variances:")
print(result_variances)

print("\nCluster Probabilities:")
print(result_probs)
