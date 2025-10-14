import random

import numpy as np
import pandas
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler

import constants
import torch
from torch.utils.data import TensorDataset, DataLoader


class DatasetInterface:
    """
    Interface class to abstract all dataset details.
    """
    def __init__(self):
        (x_train, y_train, y_train_multi,
         x_val, y_val, y_val_multi,
         x_test, y_test, y_test_multi, cat_encoder) = load_dataset()
        self.data = {'train': {'x': x_train, 'y': y_train, 'y_multi': y_train_multi},
                     'val': {'x': x_val, 'y': y_val, 'y_multi': y_val_multi},
                     'test': {'x': x_test, 'y': y_test, 'y_multi': y_test_multi}}
        self.cat_encoder = cat_encoder
        self.feature_size = x_train.shape[1]
        self.multiclass_num = np.unique(y_train_multi).shape[0]

    def info(self):
        return (f'\tTraining samples: {self.data['train']['x'].shape[0]};\n'
                f'\tValidation samples: {self.data['val']['x'].shape[0]};\n'
                f'\tTest samples: {self.data['test']['x'].shape[0]}\n'
                f'\tFeatures: {self.data['train']['x'].shape[1]}')

    # methods returning nparray data for scikit-learn
    def trainset(self, multiclass=False):
        return self.get_data('train', multiclass)

    def valset(self, multiclass=False):
        return self.get_data('val', multiclass)

    def testset(self, multiclass=False):
        return self.get_data('test', multiclass)

    # methods returning dataloaders for pytorch
    def trainloader(self, multiclass=False, bs=constants.batch_size):
        return self.get_dataloader('train', multiclass, bs)

    def valloader(self, multiclass=False, bs=constants.batch_size):
        return self.get_dataloader('val', multiclass, bs)

    def testloader(self, multiclass=False, bs=constants.batch_size):
        return self.get_dataloader('test', multiclass, bs)

    # helper functions
    def get_dataloader(self, type, multiclass=False, bs=constants.batch_size):
        """ Used by Pytorch models """
        data = self.data[type]
        y = data['y_multi'] if multiclass else data['y']
        x = torch.from_numpy(data['x']).float()
        y = torch.from_numpy(y).float()
        if multiclass:
            # cross_entropy_loss for multiclass expects shape (N, ), where N is the batch size
            # so we remove extra 1 dimensions. BCELoss requires (N, 1), so we do not squeeze() if we do binary
            # cross_etnropy_loss also needs longs as targets
            y = y.squeeze().long()

        do_shuffle = type != 'test'
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=bs, shuffle=do_shuffle)

    def get_data(self, type, multiclass=False):
        """ Used by scikit-learn models """
        data = self.data[type]
        if multiclass:
            return data['x'], data['y_multi']
        else:
            # remove extra dimensions from y so sklearn doesn't complain
            return data['x'], data['y'].squeeze()


def load_dataset():
    """ loads train and test data into pandas dataframe.
        Returns:
            train_x: normalized numpy array of floats. Categorical features encoded
            train_y: numpy array of floats. Binary feature, 0 or 1 ('label')
            train_y_multi: numpy array of floats. Categorical label encoded ('attack')
            val_x: normalized numpy array of floats. Categorical features encoded
            val_y: numpy array of floats. Binary feature, 0 or 1 ('label')
            val_y_multi: numpy array of floats. Categorical label encoded ('attack')
            test_x: normalized numpy array of floats. Encoded categorical features
            test_y: numpy array of floats. Binary feature, 0 or 1 ('label')
            test_y_multi: numpy array of floats. Encoded categorical label ('attack')
            cat_encoder: fit sklearn categorical encoder
    """
    # load train and test set into Pandas DataFrame
    df_test = pd.read_csv(".\\KDDTest.csv", header=0).drop(columns=["attack score"])
    df_train = pd.read_csv(".\\KDDTrain.csv", header=0).drop(columns=["attack score"])

    print(df_test.head(3))
    print(df_train.head(3))
    # TODO

    # encode categorical features ['protocol_type', 'service', 'flag', 'attack'] and update train/test dataframes
    categs_feats = df_train[['protocol_type', 'service', 'flag', 'attack']]
    encoder = OrdinalEncoder()
    categs_encoded = encoder.fit_transform(categs_feats)
    df_train[['protocol_type', 'service', 'flag', 'attack']] = pd.DataFrame(categs_encoded, columns=categs_feats.columns)


    categs_feats = df_test[['protocol_type', 'service', 'flag', 'attack']]
    encoder = OrdinalEncoder()
    categs_encoded = encoder.fit_transform(categs_feats)
    df_test[['protocol_type', 'service', 'flag', 'attack']] = pd.DataFrame(categs_encoded, columns=categs_feats.columns)

    # TODO

    # split 10% of training data into validation data. val_indices are the indices of the randomly selected 10%
    val_indices = random.sample(range(df_train.shape[0]), int(df_train.shape[0] * 0.1))
    # TODO

    # split features and labels. Labels are: ['attack', 'attack score', 'label'].
    train_features = df_train.drop(columns=['attack', 'label'])
    train_labels = df_train[["attack","label"]]

    test_features = df_test.drop(columns=['attack', 'label'])
    test_labels = df_test[["attack","label"]]

    # normalize features with an sklearn scaler
    scaler = StandardScaler()

    train_features = scaler.fit_transform(train_features)

    x_train = np.delete(train_features, val_indices, axis=0)
    y_train = train_labels["label"].drop(val_indices).to_numpy()
    y_train_multi = train_labels["attack"].drop(val_indices).to_numpy()

    x_val = train_features[val_indices]
    y_val = train_labels["label"][val_indices].to_numpy()
    y_val_multi = train_labels["attack"][val_indices].to_numpy()

    x_test = scaler.transform(test_features)
    y_test = test_labels["label"].to_numpy()
    y_test_multi = test_labels["attack"].to_numpy()

    # TODO

    cat_encoder = encoder
    # return train, validation, test data, and the fit categorical encoder. Datasets must be nparray
    return (x_train, y_train, y_train_multi,
                x_val, y_val, y_val_multi,
                x_test, y_test, y_test_multi,
                cat_encoder)

load_dataset()