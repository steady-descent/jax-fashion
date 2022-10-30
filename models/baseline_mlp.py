# Adapted from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
from jax import random


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def baseline_mlp(
    layer_sizes=[784, 512, 512, 10],
):
    return init_network_params(layer_sizes, random.PRNGKey(0))
