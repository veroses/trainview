from .models import Training_Request
from .scratch_nn import *
from .data import get_mnist_loaders
import numpy as np

def train_model(params : Training_Request, training_status):
    epochs = params.epochs
    lr = params.learning_rate
    batch_size = params.batch_size
    momentum = params.momentum
    beta1 = params.beta1
    beta2 = params.beta2
    optimizer = params.optimizer.lower()
    layers = [Convolution(1, 4, (3, 3), (1, 1)), Relu(), Pooling((2, 2)), Flatten(), Linear(784, 10), SoftMax()]

    if optimizer == "sgd":
        if momentum:
            small_net = Network(layers, SGD, learning_rate=lr, momentum=momentum)
        else:
            small_net = Network(layers, SGD, learning_rate=lr)
    elif optimizer == "adam":
        if beta1 and beta2:
            small_net = Network(layers, Adam, learning_rate=lr, beta1=beta1, beta2=beta2)
        elif beta1:
            small_net = Network(layers, Adam, learning_rate=lr, beta1=beta1)
        elif beta2:
            small_net = Network(layers, Adam, learning_rate=lr, beta2=beta2 )
        else:
            small_net = Network(layers, Adam, learning_rate=lr)
    else:
        print("not a valid optimizer, please try again")
        return training_status
    
    train_dataloader, test_dataloader = get_mnist_loaders(batch_size=batch_size)
    

    for epoch in range(epochs):
        loss = 0
        correct = 0
        total_count = 0
        for batch in train_dataloader:
            batch_inputs, batch_labels = batch
            

            inputs = batch_inputs.numpy()
            labels = batch_labels.numpy()
            one_hot_labels = np.eye(10)[labels]

            loss += small_net.update(inputs, one_hot_labels)

        num_batches = len(train_dataloader)
        loss /= num_batches

        for batch in test_dataloader:
            test_inputs, test_labels = batch

            test_inputs = test_inputs.numpy()
            test_labels = test_labels.numpy()
            one_hot_test_labels = np.eye(10)[test_labels]
            correct += small_net.evaluate(test_inputs, one_hot_test_labels)
            total_count += len(test_labels)

        accuracy = correct / total_count

        training_status.epoch = epoch
        training_status.loss = loss
        training_status.accuracy = accuracy

