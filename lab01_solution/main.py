import matplotlib.pyplot as plt
import sklearn.tree
import torch

from sklearn_pytorch_intro.neural_network import NeuralNetwork
from sklearn_pytorch_intro.dataset import DatasetInterface
from sklearn_pytorch_intro.neural_network import get_torch_device


def plot_loss_acc(train_history, multiclass=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr in train_history:
        for idx, (plot_type, history) in enumerate(train_history[lr].items()):
            axes[idx].set_title(f'{"Multiclass" if multiclass else "Binary"} - {plot_type} History')
            val_loss_history, train_loss_history = history['val'], history['train']
            axes[idx].plot(*list(zip(*train_loss_history)), label=f'Train {plot_type}; lr = {lr}')
            axes[idx].plot(*list(zip(*val_loss_history)), label=f'Validation {plot_type}; lr = {lr}')
            axes[idx].legend(loc='best')
    plt.show()


def build_nn_model(feature_size, class_num=1):
    nn_shape = [feature_size, 128, 64, class_num]
    nn_activations = ['relu', 'relu', 'sigmoid' if class_num == 1 else 'identity']
    optimizer = torch.optim.Adam
    model = NeuralNetwork(shape=nn_shape, activations=nn_activations, optimizer=optimizer)
    return model


def train_nn_models(dataset, learning_rates, epochs, val_interval, class_num=1):
    # used to store training loss and accuracy for each learning rate used
    train_history = {}
    # used to store the trained model for each learning rate used
    models = {}
    device = get_torch_device()
    trainloader = dataset.trainloader(multiclass=class_num != 1, bs=128)
    valloader = dataset.valloader(multiclass=class_num != 1, bs=264)
    for lr in learning_rates:
        # initialize the model with given shape, activations and learning rate
        model = build_nn_model(dataset.feature_size, class_num)
        model.to(device)
        # train (fit) the model using
        train_history[str(lr)] = model.fit(trainloader, valloader, lr, epochs, val_interval=val_interval)
        models[str(lr)] = model

    return models, train_history


def main():
    learning_rates = [0.001]
    val_interval = 1
    epochs = 5
    # get DatasetInterface object
    dataset = DatasetInterface()
    print(f'Dataset loaded.\n{dataset.info()}')

    # Traditional ML models
    train_x, train_y = dataset.trainset()
    test_x, test_y = dataset.testset()
    _, train_y_multi = dataset.trainset(multiclass=True)
    _, test_y_multi = dataset.testset(multiclass=True)
    dt_model_binary = sklearn.tree.DecisionTreeClassifier()
    dt_model_multi = sklearn.tree.DecisionTreeClassifier()
    print('Fitting Decision Tree Classifiers')
    dt_model_binary.fit(train_x, train_y)
    dt_model_multi.fit(train_x, train_y_multi)
    print(f'Decision Tree Binary Test Accuracy: {dt_model_binary.score(test_x, test_y)}')
    print(f'Decision Tree Multi Test Accuracy: {dt_model_multi.score(test_x, test_y_multi)}')

    rf_model_binary = sklearn.ensemble.RandomForestClassifier()
    rf_model_multi = sklearn.ensemble.RandomForestClassifier()
    print('Fitting Random Forest Classifiers')
    rf_model_binary.fit(train_x, train_y)
    rf_model_multi.fit(train_x, train_y_multi)
    print(f'Random Forest Binary Test Accuracy: {rf_model_binary.score(test_x, test_y)}')
    print(f'Random Forest Multi Test Accuracy: {rf_model_multi.score(test_x, test_y_multi)}')

    # DNN models
    nn_models = {}
    # train one DNN (binary) for each learning rate specified
    nn_models['binary'], binary_train_log = train_nn_models(dataset, learning_rates, epochs, val_interval, class_num=1)
    # train one DNN (multiclass) for each learning rate specified
    nn_models['multiclass'], multi_train_log = train_nn_models(dataset, learning_rates, epochs, val_interval,
                                                               class_num=dataset.multiclass_num)

    # plotting
    plot_loss_acc(binary_train_log, multiclass=False)
    plot_loss_acc(multi_train_log, multiclass=True)

    # testing
    for model_type, models in nn_models.items():
        print(f'\n{model_type.capitalize()} Models performance:')
        testloader = dataset.testloader(multiclass=not list(models.values())[0].is_binary)
        for lr in models:
            accuracy = models[lr].test(testloader)
            print(f'Learning rate {lr} test accuracy: {accuracy}')


if __name__ == '__main__':
    main()
