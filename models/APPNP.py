#!/usr/bin/env python
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import math
class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


#APPNP
class APPNP1(nn.Module):
    def __init__(self, input_dim, support,num_hidden_conv, num_classes, dropout_rate=0, encoder_type=0,second_hidn=0,K=10, alpha=0.1,tau = 0.4):
        super(APPNP1, self).__init__()
        self.tau=tau
        self.Linear1 = Linear(input_dim, num_hidden_conv, 0, bias=False)
        self.Linear2 = Linear(num_hidden_conv, num_classes, 0, bias=False)
        self.alpha = alpha
        self.K = K
        self.fc1 = torch.nn.Linear(num_hidden_conv, num_hidden_conv)
        self.fc2 = torch.nn.Linear(num_hidden_conv, num_hidden_conv)
    def forward(self, x, adj,encoder_type=0):
        h0 = torch.relu(self.Linear1(adj[0]))
        x_cl = h0
        for _ in range(self.K):
            x_cl = (1 - self.alpha) * torch.matmul(adj[0], x_cl) + self.alpha * h0
        h = self.Linear2(x_cl)
        return torch.log_softmax(h, dim=-1), x_cl
        
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
 #init net       
class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha):
        super(APPNP1, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=False)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=False)
        self.alpha = alpha
        self.K = K

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(x))
        h0 = self.Linear2(x)
        h = h0
        for _ in range(self.K):
            h = (1 - self.alpha) * torch.matmul(adj, h) + self.alpha * h0
        return torch.log_softmax(h, dim=-1)