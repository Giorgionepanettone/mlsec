import random

import numpy as np
import pandas
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler

import constants
import torch
from torch.utils.data import TensorDataset, DataLoader


class DatasetInterface:
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

        def trainset(self, multiclass=False):
            return self.get_data('train', multiclass)

        def valset(self, multiclass=False):
            return self.get_data('val', multiclass)

    def trainset(self, multiclass=False):
        return self.get_data('train', multiclass)

    def testset(self, multiclass=False):
        return self.get_data('test', multiclass)

    def trainloader(self, multiclass=False, bs=constants.batch_size):
        return self.get_dataloader('train', multiclass, bs)

    def valloader(self, multiclass=False, bs=constants.batch_size):
        return self.get_dataloader('val', multiclass, bs)

    def testloader(self, multiclass=False, bs=constants.batch_size):
        return self.get_dataloader('test', multiclass, bs)

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
    train_df = pandas.read_csv(constants.data_dir / 'KDDTrain.csv')
    test_df = pandas.read_csv(constants.data_dir / 'KDDTest.csv')

    # encode categorical features
    categ_feats = train_df[['protocol_type', 'service', 'flag', 'attack']]
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    cat_encoder.fit(categ_feats)
    encoded_feats = cat_encoder.transform(categ_feats) # transforms into np array
    train_df[['protocol_type', 'service', 'flag', 'attack']] = pd.DataFrame(encoded_feats, columns=categ_feats.columns, index=categ_feats.index)
    # cat_encoder2 =  OrdinalEncoder()
    categ_feats = test_df[['protocol_type', 'service', 'flag', 'attack']]
    encoded_feats = cat_encoder.transform(categ_feats) # transforms into np array
    test_df[['protocol_type', 'service', 'flag', 'attack']] = pd.DataFrame(encoded_feats, columns=categ_feats.columns, index=categ_feats.index)

    # split 10% of training data into validation data
    val_indices = random.sample(range(train_df.shape[0]), int(train_df.shape[0] * 0.1))
    val_df = train_df.iloc[val_indices]
    # remove validation indices from training data (axis=0)
    train_df = train_df.drop(val_indices, axis=0)

    # split features and lables
    x_train_df = train_df.drop(columns=['attack', 'attack score', 'label'])
    x_val_df = val_df.drop(columns=['attack', 'attack score', 'label'])
    x_test_df = test_df.drop(columns=['attack', 'attack score', 'label'])
    y_train = train_df.loc[:, train_df.columns == 'label'].to_numpy()
    y_train_multi = train_df.loc[:, train_df.columns == 'attack'].to_numpy()
    y_val = val_df.loc[:, val_df.columns == 'label'].to_numpy()
    y_val_multi = val_df.loc[:, train_df.columns == 'attack'].to_numpy()
    y_test = test_df.loc[:, test_df.columns == 'label'].to_numpy()
    y_test_multi = test_df.loc[:, test_df.columns == 'attack'].to_numpy()

    # normalize features
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_df)
    x_val = scaler.transform(x_val_df)
    x_test = scaler.transform(x_test_df)
    return (x_train, y_train, y_train_multi,
            x_val, y_val, y_val_multi,
            x_test, y_test, y_test_multi,
            cat_encoder)
