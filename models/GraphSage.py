import scipy.sparse as sp 
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import math
#https://zhuanlan.zhihu.com/p/391054539
 
        
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, support,num_hidden_conv, num_classes, dropout_rate=0, encoder_type=0,second_hidn=0,K=10, alpha=0.1,tau = 0.4):
        super(GraphSAGE, self).__init__()
        
        self.sage1 = SAGEConv(input_dim, num_hidden_conv)  
        self.sage2 = SAGEConv(num_hidden_conv, num_classes)
        self.tau=tau
        
        self.fc1 = torch.nn.Linear(num_hidden_conv, num_hidden_conv)
        self.fc2 = torch.nn.Linear(num_hidden_conv, num_hidden_conv)
    def forward(self, x,adj,encoder_type=0):
        
        import ipdb;ipdb.set_trace()
        x = adj[0]
        edge_index,_ = dense_to_sparse(adj[0].cpu()) 
        edge_index = torch.LongTensor(edge_index.numpy()).cuda()
        x = self.sage1(x, edge_index)
        x_cl = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1),x_cl
        
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def cl_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask ,mean_type=0):
        e = torch.eye(z1.shape[0])
        one = torch.ones(z1.shape[0],1)
        e = e.cuda()
        one = one.cuda()
        mask = mask.cuda()
        s_value = torch.mm(z1 , z1.t()).cuda()
        b_value = torch.mm(z1 , z2.t()).cuda()
        s_value = torch.exp(s_value / self.tau)
        b_value = torch.exp(b_value / self.tau)
        
        p = torch.mm(mask, one)
        p = 2 * p - 1
        p = 1/p
        
        value_mu = torch.mm(s_value, one) + torch.mm(b_value, one) - torch.mm(s_value * e, one)
        value_zi = torch.mm(s_value * (mask - e), one) + torch.mm(b_value * mask, one)
        #
        if mean_type == 0:
            loss = -torch.log(p * value_zi / value_mu)
        else:
            loss = -p * torch.log(value_zi/value_mu)
        return loss
    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, doc_mask,mask,sample_mask,sample_type=1,
             mean: bool = True, batch_size: int = 1):    
        z1 = z1[doc_mask]
        z2 = z2[doc_mask]
        if sample_type == 1:
            z1 = z1[sample_mask]
            z2 = z2[sample_mask]
            mask = mask[sample_mask]
            mask = mask[:,sample_mask]
            #print(mask.size())
            #import ipdb;ipdb.set_trace()
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        if batch_size == 0:       
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.cl_loss(h1, h2, mask)
            l2 = self.cl_loss(h2, h1, mask)
            
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        
        return ret