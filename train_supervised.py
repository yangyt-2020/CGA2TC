from __future__ import division
from __future__ import print_function
from sklearn import metrics
import random
import time
import sys
import os
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx,dense_to_sparse
from torch_geometric.utils import dropout_adj, degree, to_undirected
import torch
import torch.nn as nn
import numpy as np

from utils.utils import *
from models.gcn import GCN
from models.mlp import MLP
from models.APPNP import APPNP1
from models.GraphSage import GraphSAGE
from sklearn.manifold import TSNE
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CONFIG
cfg = CONFIG()
seed = random.randint(1, 200)
seed = 2019
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(seed)
if len(sys.argv) != 2:
	sys.exit("Use: python train.py <dataset>")
datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr','TREC','WebKB','ag_news']
dataset = sys.argv[1]
if dataset not in datasets:
	sys.exit("wrong dataset name")
cfg.dataset = dataset

param = {
        'drop_edge_rate_1': 0.2,
        'drop_edge_rate_2': 0.6,
        'drop_scheme': cfg.param,
}

# 
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)

start = time.time()
#Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
      cfg.dataset)
features = sp.identity(features.shape[0])  # featureless
# Some preprocessing
features = preprocess_features(features)
if cfg.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif cfg.model == 'APPNP':
    support = [preprocess_adj(adj)]
    model_func = APPNP1
elif cfg.model == 'GAT':
    support = [preprocess_adj(adj)]
    model_func = spGAT
elif cfg.model == 'GraphSage':
    support = [preprocess_adj(adj)]
    model_func = GraphSAGE  
elif cfg.model == 'GRU':
    support = [preprocess_adj(adj)]
    model_func = GRU   
elif cfg.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, cfg.max_degree)
    num_supports = 1 + cfg.max_degree
    model_func = GCN
elif cfg.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(cfg.model))
# Define placeholders
t_features = torch.from_numpy(features)
t_y_train = torch.from_numpy(y_train)
t_y_val = torch.from_numpy(y_val)
t_y_test = torch.from_numpy(y_test)
t_train_mask = torch.from_numpy(train_mask.astype(np.float32)).cuda()
tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1]).cuda()
t_val_mask = torch.from_numpy(np.array(val_mask * 1., dtype=np.float32)).cuda()
tm_val_mask = torch.transpose(torch.unsqueeze(t_val_mask, 0), 1, 0).repeat(1, t_y_val.shape[1]).cuda()

t_test_mask = torch.from_numpy(np.array(test_mask * 1., dtype=np.float32)).cuda()
tm_test_mask = torch.transpose(torch.unsqueeze(t_test_mask, 0), 1, 0).repeat(1, t_y_test.shape[1]).cuda()

doc_mask = t_train_mask + t_val_mask + t_test_mask 
doc_mask = doc_mask > 0
doc_mask = doc_mask.cuda()
train_labels_all = torch.max(t_y_train, 1)[1]
val_labels_all = torch.max(t_y_val, 1)[1]
test_labels_all = torch.max(t_y_test, 1)[1]

train_labels_all = train_labels_all.cuda()
val_labels_all = val_labels_all.cuda()
test_labels_all = test_labels_all.cuda()

train_labels = train_labels_all[t_train_mask.cpu() > 0]
val_labels = val_labels_all[t_val_mask.cpu() > 0]
test_labels = test_labels_all[t_test_mask.cpu() > 0]
train_labels_all = train_labels_all.contiguous().view(-1, 1)


if cfg.single_positive_example == 0:
    mask_all = torch.zeros(len(doc_mask), len(doc_mask))
    train_labels = train_labels.contiguous().view(-1, 1)
    mask = torch.eq(train_labels, train_labels.T).float().cuda()  # train labels as pos/neg examples
    mask_all[:len(mask),:len(mask)]=mask
    
    mask_train = mask_all[doc_mask]  
    mask_train = mask_train[:, doc_mask] 
    #mask_train = torch.eye(len(mask_train))
    mask_train = torch.max(mask_train, torch.eye(len(mask_train)))  
    #import ipdb;ipdb.set_trace()
t_support = []
for i in range(len(support)):
    t_support.append(torch.Tensor(support[i]).cuda())
if dataset=="20ng":#OOM
    edge_index_self = dense_to_sparse(t_support[0].cpu())
else:
    edge_index_self = dense_to_sparse(t_support[0])
if param['drop_scheme'] == 'degree':
    drop_weights = degree_drop_weights(edge_index_self[0])
    drop_weights = drop_weights.cuda()
elif param['drop_scheme'] == 'pr':
    drop_weights = pr_drop_weights(edge_index_self[0], aggr='sink', k=200)
    drop_weights = drop_weights.cuda()
else:
    drop_weights = None
t_y_train = t_y_train.cuda()
t_y_val = t_y_val.cuda()
t_y_test = t_y_test.cuda()
def drop_edge(idx: int):
        global drop_weights
        if param['drop_scheme'] == 'uniform':
            temp = dropout_adj(edge_index_self[0],edge_attr=edge_index_self[1], p=param[f'drop_edge_rate_{idx}'])
            support = [to_dense_adj(temp[0],edge_attr=temp[1])[0].cuda()]
            return support
        elif param['drop_scheme'] in ['degree', 'pr']:
            support = drop_edge_weighted(edge_index_self, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=cfg.threshold)
            return support
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')
def evaluate(model,criterion,features, labels, mask):
   t_test = time.time()
   model.eval()
   with torch.no_grad():
       logits,_ = model(t_features,t_support,encoder_type=cfg.encoder_type)
       t_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32)).cuda()
       tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
       loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
       pred = torch.max(logits, 1)[1]
       acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()
       test_mask_logits = logits * tm_mask
       test_label = labels *tm_mask
   return loss.cpu().numpy(), acc, pred.cpu().numpy(), labels.cpu().numpy(), (time.time() - t_test),test_mask_logits.cpu().numpy(),test_label.cpu().numpy()


model = model_func(input_dim=features.shape[0],support=t_support, num_hidden_conv=cfg.hidden1,dropout_rate=cfg.dropout,encoder_type=cfg.encoder_type,second_hidn=cfg.second_hidn,num_classes=y_train.shape[1])
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
# Define model evaluation function
#Train model
for epoch in range(cfg.epochs):
    model.train()
    t = time.time()
    logits,_ = model(t_features,t_support,encoder_type=cfg.encoder_type)
    loss_CE = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
    if epoch<cfg.pre_epoch or cfg.CL_weight ==0:
        loss = loss_CE
    else:
        support1 = drop_edge(1)
        support2 = drop_edge(2)
        _,represention1= model(t_features, support1,encoder_type=cfg.encoder_type)
        _,represention2= model(t_features, support2,encoder_type=cfg.encoder_type)

        sample_mask = torch.empty(int(sum(doc_mask)),dtype=torch.float32).uniform_(0,1).cuda()
        sample_mask = sample_mask <= cfg.sample_size

        loss_CL = model.loss(represention1, represention2 ,doc_mask,mask_train,sample_mask,cfg.sample_type)
        loss = loss_CE + cfg.CL_weight * loss_CL

    acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    val_loss, val_acc, pred, labels, duration,_ ,_= evaluate(model,criterion,t_features, t_y_val, val_mask)
    print_log("Epoch: {:3.0f} t_loss= {:.5f} t_acc= {:.4f} v_loss= {:.5f} v_acc= {:.5f} time= {:.4f}"\
                .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

print_log("Optimization Finished!")
test_loss, test_acc, pred, labels, test_duration,test_mask_logits,test_label = evaluate(model,criterion,t_features, t_y_test, test_mask)
print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc, test_duration))









