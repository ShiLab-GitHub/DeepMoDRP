import csv
import torch.nn as nn
from sklearn import preprocessing
import torch
import torch.utils.data as Data
from scipy.stats import pearsonr
import pickle


EPOCH = 500
BATCH_SIZE = 80
LR = 1e-4


def read_cell_line_list(filename):  # load cell lines and build a dictionary
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    cell_line_dict = {}
    index = 0
    for line in reader:
        cell_line_dict[line[0]] = index
        index += 1
    return cell_line_dict


def read_cell_line_copynumber(filename, cell_line_dict):  # load one of the features of cell line - copynumber
    f = open(filename, 'r')
    reader = csv.reader(f)
    for i in range(1):
        reader.__next__()
    copynumber = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            copynumber[cell_line_dict[line[0]]] = line[1:]
    return copynumber

#Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        x_noisy = x + 0.1 * torch.randn_like(x)  # add noise
        x_encoded = self.encoder(x_noisy)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded

# Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded

class CombinedModel(nn.Module):
    def __init__(self, denoising_ae, sparse_ae):
        super(CombinedModel, self).__init__()
        self.denoising_ae = denoising_ae
        self.sparse_ae = sparse_ae

    def forward(self, x):
        x_denoising_ae, x_denoising_dae= self.denoising_ae(x)
        x_sparse_ae, x_sparse_dae= self.sparse_ae(x)
        x_combined = torch.cat([x_denoising_ae, x_sparse_ae], dim=1)
        x_combined_decoder =x_denoising_dae + x_sparse_dae
        return x_combined, x_combined_decoder

def load_data():
    cell_line_dict = read_cell_line_list('../data/cellline/cellline_listwithACH_80cellline.csv')
    copynumber = read_cell_line_copynumber('../data/cellline/CNV_85dim_23430dim.csv', cell_line_dict)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    copynumber = min_max_scaler.fit_transform(copynumber)
    train_data = torch.FloatTensor(copynumber)

    train_size = int(train_data.shape[0] * 0.8)
    data_train = train_data[:train_size]
    data_test = train_data[train_size:]

    return data_train, data_test, train_data


def train(train_data, test_data, data_all):
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    input_size = 24254
    hidden_size = 256
    denoising_ae = DenoisingAutoencoder(input_size, hidden_size)
    sparse_ae = SparseAutoencoder(input_size, hidden_size)

    combined_model = CombinedModel(denoising_ae, sparse_ae)
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    best_loss = 1
    res = [[0 for col in range(512)] for row in range(80)]

    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader):
            encoded, decoded = combined_model(data)
            loss = loss_func(decoded, data)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

        test_en, test_de = combined_model(test_data)
        test_loss = loss_func(test_de, test_data)
        pearson = pearsonr(test_de.view(-1).tolist(), test_data.view(-1))[0]
        if test_loss < best_loss:
            best_loss = test_loss
            res, _ = combined_model(data_all)
            pickle.dump(res.data.numpy(), open('../data/cellline/512dim_copynumber.pkl', 'wb'))
            print("best_loss: ", best_loss.data.numpy())
            print("pearson: ", pearson)

    return


if __name__ == "__main__":
    train_data, test_data, all_data = load_data()
    train(train_data, test_data, all_data)
