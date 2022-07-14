import time
import jax.numpy as jnp
from data_utils import NumpyLoader, load_fashion_mnist
from models.baseline_nn import baseline_nn
from models.model_utils import accuracy, one_hot, update


if __name__ == "__main__":

    step_size = 0.01
    num_epochs = 8
    batch_size = 128
    n_targets = 10

    params = baseline_nn()
    fashion_dataset = load_fashion_mnist()
    training_generator = NumpyLoader(
        fashion_dataset, batch_size=batch_size, num_workers=0
    )

    # Get the full train dataset (for checking accuracy while training)
    train_images = jnp.array(fashion_dataset.data).reshape(
        len(fashion_dataset.data), -1
    )
    train_labels = one_hot(jnp.array(fashion_dataset.targets), n_targets)
    # Get full test dataset
    fashion_dataset_test = load_fashion_mnist(is_train=False)
    test_images = jnp.array(fashion_dataset_test.data).reshape(
        len(fashion_dataset_test.data), -1
    )
    test_labels = one_hot(jnp.array(fashion_dataset_test.targets), n_targets)

    # `batched_predict` has the same call signature as `predict`
    train_acc = accuracy(params, train_images, train_labels)
    print("train acc before training", train_acc)

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, n_targets)
            params = update(params, step_size, x, y)
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
