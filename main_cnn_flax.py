# partially based on https://flax.readthedocs.io/en/latest/getting_started.html
import time
import jax.numpy as jnp
import jax
from data_utils import NumpyLoader, load_fashion_mnist
from models.baseline_cnn_flax import CNN
from models.model_utils import update
from tqdm import tqdm
import optax


def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)
    init_rngs = {
        "params": jax.random.PRNGKey(0),
        "dropout": jax.random.PRNGKey(1),
    }

    step_size = 0.001
    num_epochs = 8
    batch_size = 256
    n_targets = 10

    # params = baseline_nn()
    cnn = CNN()
    params = cnn.init(init_rngs, jnp.ones([1, 28, 28, 1]))["params"]
    fashion_dataset = load_fashion_mnist()
    training_generator = NumpyLoader(
        fashion_dataset, batch_size=batch_size, num_workers=0
    )

    # Get the full train dataset (for checking accuracy while training)
    train_images = jnp.array(fashion_dataset.data).reshape(
        len(fashion_dataset.data), 28, 28, 1
    )
    train_labels = jnp.array(fashion_dataset.targets)
    # Get full test dataset
    fashion_dataset_test = load_fashion_mnist(is_train=False)
    test_images = jnp.array(fashion_dataset_test.data).reshape(
        len(fashion_dataset_test.data), 28, 28, 1
    )
    test_labels = jnp.array(fashion_dataset_test.targets)

    # `batched_predict` has the same call signature as `predict`
    logits = CNN().apply(
        {"params": params}, train_images, rngs={"dropout": jax.random.PRNGKey(2)}
    )
    print(train_images.shape)
    metrics = compute_metrics(logits=logits, labels=train_labels)
    print("train acc before training", metrics["accuracy"])

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in tqdm(list(training_generator)):
            labels_onehot = jax.nn.one_hot(y, num_classes=10)

            def loss_fn(params):
                logits = CNN().apply({"params": params}, x.reshape(-1, 28, 28, 1), rngs={"dropout": jax.random.PRNGKey(2)})
                loss = cross_entropy_loss(logits=logits, labels=y)
                return loss, logits

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, logits), grads = grad_fn(params)
            params = update(params, step_size, grads)

        epoch_time = time.time() - start_time

        train_results = compute_metrics(
            logits=CNN().apply({"params": params}, train_images, rngs={"dropout": jax.random.PRNGKey(2)}), labels=train_labels
        )
        test_results = compute_metrics(
            logits=CNN().apply({"params": params}, test_images, rngs={"dropout": jax.random.PRNGKey(2)}), labels=test_labels
        )
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_results["accuracy"]))
        print("Test set accuracy {}".format(test_results["accuracy"]))
