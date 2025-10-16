import matplotlib.pyplot as plt
import sklearn.tree
import torch

#from sklearn_pytorch_intro.neural_network import NeuralNetwork
from dataset import DatasetInterface
#from sklearn_pytorch_intro.neural_network import get_torch_device
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def plot_loss_acc(train_history, multiclass=False):
    """ plots training and validation loss and accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr in train_history:
        for idx, (plot_type, history) in enumerate(train_history[lr].items()):
            axes[idx].set_title(f'{"Multiclass" if multiclass else "Binary"} - {plot_type} History')
            val_loss_history, train_loss_history = history['val'], history['train']
            axes[idx].plot(*list(zip(*train_loss_history)), label=f'Train {plot_type}; lr = {lr}')
            axes[idx].plot(*list(zip(*val_loss_history)), label=f'Validation {plot_type}; lr = {lr}')
            axes[idx].legend(loc='best')
    plt.show()


def main():
    # set some values here to test out different epochs/lr. val_interval is how often we evaluate on validation data
    # pytorch-only
    learning_rates = [0.001, 0.01]
    epochs = 5
    val_interval = 1

    # get DatasetInterface object
    dataset = DatasetInterface()
    print(f'Dataset loaded.\n{dataset.info()}')

    # Create and train traditional ML models: DecisionTree, RandomForest (others if you want)
    # TODO decision tree setup
    train_x, train_y = dataset.trainset()
    _, train_y_multi = dataset.trainset(multiclass=True)
    test_x, test_y = dataset.testset()
    _, test_y_multi = dataset.testset(multiclass=True)

    print('Fitting Decision Tree Classifiers')
    # TODO decision tree training
    dt_model_binary = DecisionTreeClassifier(random_state=42)
    dt_model_multi = DecisionTreeClassifier(random_state=42)
    dt_model_binary.fit(train_x, train_y)
    dt_model_multi.fit(train_x, train_y_multi)
    print(f'Decision Tree Binary Test Accuracy: {dt_model_binary.score(test_x, test_y)}')
    print(f'Decision Tree Multi Test Accuracy: {dt_model_multi.score(test_x, test_y_multi)}')

    # TODO random forest setup
    rf_model_binary = RandomForestClassifier(random_state=42)
    rf_model_multi = RandomForestClassifier(random_state=42)
    print('Fitting Random Forest Classifiers')
    # TODO random forest training
    rf_model_binary.fit(train_x, train_y)
    rf_model_multi.fit(train_x, train_y_multi)
    print(f'Random Forest Binary Test Accuracy: {rf_model_binary.score(test_x, test_y)}')
    print(f'Random Forest Multi Test Accuracy: {rf_model_multi.score(test_x, test_y_multi)}')
    
    # Create and train DNN models w/ pytorch
    nn_models = {}
    # create and train one DNN (binary) for each learning rate specified
    # TODO
    # store models as dictionaries {lr: model_object} in nn_models and training history as {lr: history} in binary_train_log
    # remember that NeuralNetwork.fit() returns training and validation histories
    nn_models['binary'] = ""# TODO
    binary_train_log = ""# TODO
    # create and train one DNN (multiclass) for each learning rate specified
    # TODO
    # store models as dictionaries {lr: model_object} in nn_models and training history as {lr: history} in multi_train_log
    # remember that NeuralNetwork.fit() returns training and validation histories
    nn_models['multiclass'] = ""# TODO
    multi_train_log = ""# TODO

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
