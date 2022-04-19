#!/usr/bin/env python
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support,\
                        act_func = None, \
                        featureless = False, \
                        bias=True):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))
        self.act_func = act_func
    def forward(self, x,support): 
        for i in range(len(self.support)):#1
            #import ipdb;ipdb.set_trace()
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i))) 
            if i == 0:
                out =  support[i].mm(pre_sup) 
            else:#no use
                print("else")
                out +=  support[i].mm(pre_sup)
        if self.act_func is not None:
            out = self.act_func(out)
        self.embedding = out
        return out





class GCN(nn.Module):
    def __init__( self, input_dim, \
                        support, \
                        num_hidden_conv,\
                        dropout_rate, \
                        encoder_type,\
                        second_hidn,\
                        tau = 0.4,\
                        num_classes=10):
        super(GCN, self).__init__()
        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, num_hidden_conv,support,act_func=nn.ReLU(), featureless=True)
        self.layer2 = GraphConvolution( num_hidden_conv, num_classes,support)
    
       
        
        
        self.tau = tau
        #projection head
        if encoder_type in [0,1] :
            self.fc1 = torch.nn.Linear(num_hidden_conv,num_hidden_conv)
            self.fc2 = torch.nn.Linear(num_hidden_conv, num_hidden_conv)
        else:
            self.layer3 = GraphConvolution( num_hidden_conv, second_hidn,support)#A(h1)X + W
            self.fc_classification = torch.nn.Linear(second_hidn, num_classes)#w
            self.fc1 = torch.nn.Linear(second_hidn,second_hidn)
            self.fc2 = torch.nn.Linear(second_hidn, second_hidn)
        self.dropout = nn.Dropout(dropout_rate)
 
    def forward(self, x,support,encoder_type=0):
        if encoder_type == 0 : #A(AXW) as CL
            out = self.layer1(x,support)

            first_represention = torch.mm(support[0],out)#
            out = self.dropout(out)
            out = self.layer2(out,support)
            return out,first_represention
        elif encoder_type == 1 : #AXW as CL 
            out = self.layer1(x,support)
            first_represention = out#
            out = self.dropout(out)
            out = self.layer2(out,support)
            return out,first_represention
        elif encoder_type == 2 :#A(AXW)X [as CL ]+ w (as CE)
            out = self.layer1(x,support)
            out = self.dropout(out)
            second_represention= self.layer3(out,support)

            out = F.relu(self.fc_classification(second_represention))
            return out,second_represention
            
            
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