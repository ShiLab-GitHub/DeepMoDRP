import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, GCNConv, GATConv, JumpingKnowledge, global_max_pool, global_mean_pool

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.gcn1 = GCNConv(78,self.dim_drug)
        self.gcn2 = GCNConv(self.dim_drug,self.dim_drug)
        self.relu = nn.ReLU()
        for i in range(self.layer_drug):
            if i:
               block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
               block = nn.Sequential(nn.Linear(78, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        x_drug_m = 1
        x_drug_e = 0
        x_gcn  = self.gcn1(x,edge_index)
        x_gcn1 = self.relu(x_gcn)
        x_gcn2 = self.gcn2(x_gcn1,edge_index)
        x_gcn2_ = self.relu(x_gcn2)
        x_gcn2_a = x_gcn2_+x_gcn1
        x_gcn2_m = x_gcn2_*x_gcn1
        
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_m*=x
            x_drug_e+=x
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        node_representation=torch.cat([node_representation,x_drug_m,x_drug_e,x_gcn1,x_gcn2_,x_gcn2_a,x_gcn2_m],dim=-1)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug