import torch
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from model.GNN_drug import GNN_drug
import torch.nn as nn
import torch.nn.functional as F
class DeepMoDRP_Net(torch.nn.Module):
    def __init__(self, n_filters=4, output_dim=256, dropout=0.5):

        super(MofaDRP_Net, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.layer_drug = 3
        self.dim_drug = 128
        self.dropout_ratio = 0.2
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=self.dropout_ratio)

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * (self.layer_drug+6), 1024),
            nn.ReLU(),
            # Addi
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # meth
        self.cm_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.cm_bn1 = nn.BatchNorm1d(n_filters)
        self.cm_pool1 = nn.MaxPool1d(3)
        self.cm_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.cm_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.cm_pool2 = nn.MaxPool1d(3)
        self.cm_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.cm_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.cm_pool3 = nn.MaxPool1d(3)
        self.cm_fc1 = nn.Linear(160, 512)
        self.cm_bn4 = nn.BatchNorm1d(512)
        self.cm_fc2 = nn.Linear(512, output_dim)
        self.cm_bn5 = nn.BatchNorm1d(output_dim)

        # copynumber
        self.cc_fc1 = nn.Linear(512, 1024)
        self.cc_bn1 = nn.BatchNorm1d(1024)
        self.cc_fc2 = nn.Linear(1024, 256)
        self.cc_bn2 = nn.BatchNorm1d(256)
        self.cc_fc3 = nn.Linear(256, output_dim)
        self.cc_bn3 = nn.BatchNorm1d(output_dim)
        # mut
        self.ccg_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.ccg_bn1 = nn.BatchNorm1d(n_filters)
        self.ccg_pool1 = nn.MaxPool1d(3)
        self.ccg_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.ccg_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.ccg_pool2 = nn.MaxPool1d(3)
        self.ccg_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.ccg_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.ccg_pool3 = nn.MaxPool1d(3)
        self.ccg_fc1 = nn.Linear(1136, 512)
        self.ccg_bn4 = nn.BatchNorm1d(512)
        self.ccg_fc2 = nn.Linear(512, output_dim)
        self.ccg_bn5 = nn.BatchNorm1d(output_dim)
        # RNAseq
        self.cc_fc1 = nn.Linear(512, 1024)
        self.cc_bn1 = nn.BatchNorm1d(1024)
        self.cc_fc2 = nn.Linear(1024, 256)
        self.cc_bn2 = nn.BatchNorm1d(256)
        self.cc_fc3 = nn.Linear(256, output_dim)
        self.cc_bn3 = nn.BatchNorm1d(output_dim)
        # fusion layers
        self.comb_fc1 = nn.Linear(5 * output_dim, 1024)
        self.comb_bn1 = nn.BatchNorm1d(1024)
        self.comb_fc2 = nn.Linear(1024, 128)
        self.comb_bn2 = nn.BatchNorm1d(128)
        self.comb_out = nn.Linear(128, 1)

    def forward(self, data):

        #x, edge_index, batch = data.x, data.edge_index, data.batch
        meth = data.meth
        meth = meth[:, None, :]
        copynumber = data.copynumber
        mut = data.mut
        mut = mut[:, None, :]
        RNAseq = data.RNAseq
        x_drug = self.GNN_drug(data)
        x_drug = self.drug_emb(x_drug)

        # meth
        xcm = self.cm_conv1(meth)
        xcm = self.cm_bn1(xcm)
        xcm = self.relu(xcm)
        xcm = self.cm_pool1(xcm)
        xcm = self.cm_conv2(xcm)
        xcm = self.cm_bn2(xcm)
        xcm = self.relu(xcm)
        xcm = self.cm_pool2(xcm)
        xcm = self.cm_conv3(xcm)
        xcm = self.cm_bn3(xcm)
        xcm = self.relu(xcm)
        xcm = self.cm_pool3(xcm)
        xcm = xcm.view(-1, xcm.shape[1] * xcm.shape[2])
        xcm = self.cm_fc1(xcm)
        xcm = self.cm_bn4(xcm)
        xcm = self.cm_fc2(xcm)
        xcm = self.cm_bn5(xcm)

        # cell copynumber
        xcc = self.cc_fc1(copynumber)
        xcc = self.cc_bn1(xcc)
        xcc = self.relu(xcc)
        xcc = self.cc_fc2(xcc)
        xcc = self.cc_bn2(xcc)
        xcc = self.relu(xcc)
        xcc = self.cc_fc3(xcc)
        xcc = self.cc_bn3(xcc)
        # mut
        xcg = self.ccg_conv1(mut)
        xcg = self.ccg_bn1(xcg)
        xcg = self.relu(xcg)
        xcg = self.ccg_pool1(xcg)
        xcg = self.ccg_conv2(xcg)
        xcg = self.ccg_bn2(xcg)
        xcg = self.relu(xcg)
        xcg = self.ccg_pool2(xcg)
        xcg = self.ccg_conv3(xcg)
        xcg = self.ccg_bn3(xcg)
        xcg = self.relu(xcg)
        xcg = self.ccg_pool3(xcg)
        xcg = xcg.view(-1, xcg.shape[1] * xcg.shape[2])
        xcg = self.ccg_fc1(xcg)
        xcg = self.ccg_bn4(xcg)
        xcg = self.relu(xcg)
        xcg = self.ccg_fc2(xcg)
        xcg = self.ccg_bn5(xcg)
        # RNAseq
        xcr = self.cc_fc1(RNAseq)
        xcr = self.cc_bn1(xcr)
        xcr = self.relu(xcr)
        xcr = self.cc_fc2(xcr)
        xcr = self.cc_bn2(xcr)
        xcr = self.relu(xcr)
        xcr = self.cc_fc3(xcr)
        xcr = self.cc_bn3(xcr)

        xfusion = torch.cat((x_drug, xcm, xcc, xcg, xcr), 1)
        xfusion = self.comb_fc1(xfusion)
        xfusion = self.comb_bn1(xfusion)
        xfusion = self.relu(xfusion)
        xfusion = self.dropout(xfusion)
        xfusion = self.comb_fc2(xfusion)
        xfusion = self.comb_bn2(xfusion)
        xfusion = self.relu(xfusion)
        xfusion = self.dropout(xfusion)

        out = self.comb_out(xfusion)
        out = self.sigmoid(out)
        out = out.view(-1, 1)
        return out