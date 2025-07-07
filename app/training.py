from app.models import Training_Request
from scratch_nn import *
from .data import get_mnist_loaders

def train_model(params : Training_Request, training_status):
    epochs = params.epochs
    optimizer = params.optimizer
    lr = params.learning_rate
    batch_size = params.batch_size
    momentum = params.momentum
    beta1 = params.momentum
    beta2 = params.momentum

    train_dataloader, test_dataloader = get_mnist_loaders(batch_size=batch_size)
    
    #initialize model
    layers = [Convolution(1, 4, (3, 3), (1, 1)), Relu(), Pooling((2, 2)), Relu(), Flatten(), Linear(196, 10), SoftMax]
    small_net = Network(layers, optimizer, lr) 

    for epoch in range(epochs):
        loss = 0
        correct = 0
        total_count = 0
        for batch in train_dataloader:
            batch_inputs, batch_labels = batch

            inputs = batch_inputs.numpy()
            labels = batch_labels.numpy()

            loss += small_net.update(inputs, labels)

        num_batches = len(train_dataloader)
        loss /= num_batches

        for batch in test_dataloader:
            test_inputs, test_labels = batch
            correct += small_net.evaluate(test_inputs, test_labels)
            total_count += len(test_labels)

        accuracy = correct / total_count

        training_status["epoch"] = epoch
        training_status["loss"] = loss
        training_status["accuracy"] = accuracy

