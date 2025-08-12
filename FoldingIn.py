from UltraGCN import UltraGCN, bpr_loss, mse_loss2
from Dataloder import data_loader, MovieLens1MDataset, MovieLens20MDataset
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import train_test_split

from scipy.optimize import minimize
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import heapq
from random import randrange
from random import seed as set_seed
import numpy as np
from numba import njit, prange
from pandas.api.types import is_numeric_dtype

import numpy as np
import pandas as pd
import torch.utils.data
        
        
dataset_path = "ml-1m/ratings.dat"
dataset = MovieLens1MDataset(dataset_path)
dataset.items
dataset.data
user_num = dataset.field_dims[0]
item_num = dataset.field_dims[1]
print("Number of users: ", user_num, ", Number of items: ", item_num)
columns_name=['user_id','item_id','rating', 'timestamp']
df = pd.DataFrame(dataset.data, columns = columns_name)


import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def matrix_filler(data, matrix):
    for i in tqdm(range(data.shape[0])):
        matrix[data[i][0]][data[i][1]] = 1.0
    return matrix

df = getDF('beauty/reviews_Beauty_5.json.gz')

new_df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].copy()
new_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
new_df['rating'] = 1

new_df['user_id'], unique_user_ids = pd.factorize(new_df['user_id'])

new_df['item_id'], unique_item_ids = pd.factorize(new_df['item_id'])
new_df['user_id'] += 1
new_df['item_id'] += 1
df = new_df

user_num = len(unique_user_ids)
item_num = len(unique_item_ids)
df_sorted = df.sort_values(by='timestamp')

# Рассчитываем размер тестовой выборки (3%)
test_size = int(len(df_sorted) * 0.1)
train = df_sorted.head(len(df_sorted) - test_size)
vt = df_sorted.tail(test_size)
warm_t = vt.tail(test_size // 2)
warm_v = vt.head(test_size - test_size // 2)
test = warm_t.loc[warm_t.groupby('user_id')['timestamp'].idxmax()]
val = warm_v.loc[warm_v.groupby('user_id')['timestamp'].idxmax()]
warm_test = warm_t[~warm_t.index.isin(test.index)]
warm_val = warm_v[~warm_v.index.isin(val.index)]

test_inds = test['user_id'].unique()
val_inds = val['user_id'].unique()

train.drop(columns=['rating', 'timestamp'], inplace = True)
warm_test.drop(columns=['rating', 'timestamp'], inplace = True)
warm_val.drop(columns=['rating', 'timestamp'], inplace = True)
test.drop(columns=['rating','timestamp'], inplace = True)
val.drop(columns=['rating','timestamp'], inplace = True)

train = np.array(train)
warm_test = np.array(warm_test)
warm_val = np.array(warm_val)
test = np.array(test)
val = np.array(val)
train = train.astype(np.int32) - 1
warm_test = warm_test.astype(np.int32) - 1
warm_val = warm_val.astype(np.int32) - 1
test = test.astype(np.int32) - 1
test_inds = test_inds.astype(np.int32) - 1
val = val.astype(np.int32) - 1
val_inds = val_inds.astype(np.int32) - 1

train_mas = np.zeros((user_num, item_num))
test_mas = np.zeros((user_num, item_num))
valid_mas = np.zeros((user_num, item_num))

train_matrix = torch.tensor(matrix_filler(train, train_mas))
warm_test_matrix = torch.tensor(matrix_filler(np.concatenate([np.concatenate([train, warm_val]), warm_test]), train_mas))
warm_val_matrix = torch.tensor(matrix_filler(np.concatenate([train, warm_val]), train_mas))
test_matrix = torch.tensor(matrix_filler(test, test_mas))
val_matrix = torch.tensor(matrix_filler(val, valid_mas))

train_df = pd.DataFrame(train, columns = ['user_id', 'item_id'])
warm_test_df = pd.DataFrame(warm_test, columns = ['user_id', 'item_id'])
warm_val_df = pd.DataFrame(warm_val, columns = ['user_id', 'item_id'])
test_df = pd.DataFrame(test, columns = ['user_id', 'item_id'])
val_df = pd.DataFrame(val, columns = ['user_id', 'item_id'])

#Metrics

def metrics(targets, predictions, train):
    """
    Compute metrics:
    MRR
    MRR@10
    HR@1
    HR@3
    HR@10
    Parameters:
    target - real interactions
    predictions - recomendations
    """
    inf = 1e9
    predictions = predictions - train * inf
    _, idx = torch.sort(predictions, dim=1, descending=True)

    targets_sorted = targets.gather(1, idx)
    ranks = (targets_sorted > 0.1).nonzero(as_tuple=False)[:, 1]
    ranks = ranks + 1

    mrr = torch.mean(1 / ranks)
    
    idx_10 = idx[:, :10]
    targets_sorted_10 = targets.gather(1, idx_10)
    ranks_10 = (targets_sorted_10 > 0.1).nonzero(as_tuple=False)[:, 1]
    ranks_10 = ranks_10 + 1
    mrr_10 = torch.sum(1 / ranks_10) / ranks.shape[0]
    
    cov = len(np.unique(idx_10)) / item_num

    hits = []
    for k in [1, 3, 10]:
        hits += [(ranks <= k).float().mean()]
        
    def ndcg_at_k(targets, idx_k, k):
        # Get the relevance scores for the top-k predictions
        relevance = targets.gather(1, idx_k)
        # Calculate DCG
        dcg = (relevance / torch.log2(torch.arange(2, k + 2).float())).sum(dim=1)
        # Calculate IDCG
        ideal_relevance = torch.sort(relevance, dim=1, descending=True)[0]
        idcg = (ideal_relevance / torch.log2(torch.arange(2, k + 2).float())).sum(dim=1)
        # Avoid division by zero
        idcg = torch.where(idcg == 0, torch.tensor(1e-10, device=idcg.device), idcg)
        return (dcg / idcg).mean()

    ndcg = [ndcg_at_k(targets, idx[:, :k], k) for k in [1, 3, 10]]

    return {
        "mrr" : mrr,
        "mrr@10": mrr_10,
        "hits@1": hits[0],
        "hits@3": hits[1],
        "hits@10": hits[2],
        "ndcg@1": ndcg[0],
        "ndcg@3": ndcg[1],
        "ndcg@10": ndcg[2],
        "cov": cov
    }


def convert_to_sparse_tensor(dok_mtrx):
    
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor





#Traning

latent_dim =32
n_layers = 2

lightGCN = UltraGCN(train_df, user_num, item_num, n_layers, latent_dim)
matr = lightGCN.get_A_tilda()
matr_warm = lightGCN.get_A_tilda(pd.concat([train_df, warm_val_df]))
optimizer = torch.optim.Adam(lightGCN.parameters(), lr = 0.001)
EPOCHS = 50
BATCH_SIZE = 4096
DECAY = 0.00001
K = 10
E = torch.load("LightGCN_weights_clothes.pt")

#Get Initial Metrics:

lightGCN.E0.weight = nn.Parameter(E)
print(torch.linalg.norm(E - lightGCN.E0.weight))

lightGCN.eval()
with torch.no_grad():
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers()
        final_user_Embed = final_user_Embed.cpu()
        final_item_Embed = final_item_Embed.cpu()
        initial_user_Embed = initial_user_Embed.cpu()
        initial_item_Embed = initial_item_Embed.cpu()
        
        scores = torch.matmul(final_user_Embed, torch.transpose(final_item_Embed,0, 1))
        metr_test = metrics(val_matrix[val_inds], scores[val_inds], train_matrix[val_inds])
        print(metr_test)




E = torch.load("LightGCN_weights_clothes.pt")
lightGCN.E0.weight = nn.Parameter(E)
lightGCN.to(device)

print(E.shape)

# Ours
# Fast Update by solving system of the equations

final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers_new(pd.concat([warm_val_df, train_df]), val_inds)
mas = []

U, S, Vh = torch.linalg.svd(initial_item_Embed, full_matrices = False)
Vt = Vh.transpose(0, 1) @ torch.diag(1 / S) @ U.transpose(0, 1)
Vt = Vt.cpu().detach().numpy()
train_new_matrix = warm_val_matrix 
for us in tqdm(val_inds):
    a0 = train_new_matrix[us].numpy()
    d_u = np.sum(a0)
    d_u = lightGCN.u_d[us].cpu().detach().numpy()
    b_u = np.sqrt(1 + d_u) / (d_u + 0.0001)
    b_i = (1 / torch.sqrt(1 + lightGCN.i_d)).transpose(0, 1).cpu().detach().numpy()
    b_ui = b_u * b_i
    #u = (1 /(lamb + b_ui)).T * initial_item_Embed.cpu().detach().numpy().T @ a0
    u = (1 /(lamb + b_ui)).T * Vt @ a0
    mas += [torch.tensor(u)]

with torch.no_grad():
    for ind, us in tqdm(enumerate(val_inds)):
        lightGCN.E0.weight[us] = 1 * mas[ind]

#BFGS

final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed, laplas_user, laplas_item = lightGCN.propagate_through_layers_new(pd.concat([warm_val_df, train_df]), val_inds)
mas = []
ts2 = []
ts3 = []
ts4 = []
ts5 = []
ts6 = []
ts7 = []
train_new_matrix = warm_val_matrix 
for us in tqdm(val_inds[:200]):
    t1 = time.time()
    a0 = train_new_matrix[us].numpy()
    a1 = torch.matmul(laplas_user[us, :-1].unsqueeze(0), torch.cat((initial_user_Embed.cpu(), initial_item_Embed.cpu()), 0)[:-1])
    t2 = time.time()
    d = laplas_user[us, us]
    ts2 += [t2 - t1]
    A0 = final_item_Embed.cpu() - torch.matmul(laplas_item[:, us].unsqueeze(1) / d, a1)
    b0 = laplas_item[:, us].unsqueeze(1) / d
    t3 = time.time()
    ts3 += [t3 - t2]
    A0 = A0.detach().numpy()
    b0 = b0.detach().numpy()
    t4 = time.time()
    ts4 += [t4 - t3]
    random_numbers = random.sample(range(0, final_user_Embed.cpu().shape[0]), 5)
    v_m = torch.mean(initial_user_Embed[random_numbers].cpu(), 0).detach().numpy()
    v_2 = initial_user_Embed[us].cpu().detach().numpy()
    t5 = time.time()
    ts5 += [t5 - t4]
    v_m = v_2 * 0.8 + v_m * 0.2
    t55 = time.time()
    a1 = a1.cpu().detach().numpy()
    t555 = time.time()
    t5555 = time.time()
    d = d.cpu().detach().numpy() 
    t6 = time.time()
    ts6 += [t6 - t5]

    
    def objective(v):
        v = np.array(v)
        norm_v2 = np.linalg.norm(v)**2
        term1 = np.linalg.norm(A0 @ v - a0)**2
        term2 = -2 * b0.T @ a0 * norm_v2
        term3 = 2 * norm_v2 * ((b0.T @ A0) @ v)
        term4 = norm_v2**2 * np.linalg.norm(b0)**2
        term5 = 2 * np.linalg.norm((v - a1) / d - v_m)**2
        return term1 + term2 + term3 + term4 + term5
    

    def gradient(v):
        v = np.array(v)
        norm_v2 = np.linalg.norm(v)**2
        grad_term1 = 2 * A0.T @ (A0 @ v - a0)
        grad_term2 = -4 * b0.T @ a0 * v
        grad_term3 = 4 * ((b0.T @ A0) @ v) * v + 2 * norm_v2 * (b0.T @ A0)
        grad_term4 = 4 * norm_v2 * v * np.linalg.norm(b0)**2
        grad_term5 =  4 * ((v - a1) / d - v_m) / d
        return grad_term1 + grad_term2 + grad_term3 + grad_term4 + grad_term5
    v0 = final_user_Embed[us].detach().cpu().numpy()


    result = minimize(objective, v0, jac=gradient, method='L-BFGS-B', tol = 1e-7)
    t7 = time.time()
    #print("t7:", t7 - t6)
    ts7 += [t7 - t6]
    v_opt = result.x
    mas += [(torch.tensor(v_opt - a1) / d)]
    #mas += [(torch.tensor(v_opt) - a1)]
    #break

with torch.no_grad():
    for ind, us in tqdm(enumerate(val_inds[:200])):
        lightGCN.E0.weight[us] = mas[ind] * 0.1 + lightGCN.E0.weight[us].cpu() * 0.1

print("t2:", np.mean(ts2))
print("t3:", np.mean(ts3))
print("t4:", np.mean(ts4))
print("t5:", np.mean(ts5))
print("t6:", np.mean(ts6))
print("t7:", np.mean(ts7))



lightGCN.eval()
with torch.no_grad():
    
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers()
        final_user_Embed = final_user_Embed.cpu()
        final_item_Embed = final_item_Embed.cpu()
        initial_user_Embed = initial_user_Embed.cpu()
        initial_item_Embed = initial_item_Embed.cpu()
        
        scores = torch.matmul(final_user_Embed, torch.transpose(final_item_Embed,0, 1))
        metr_test = metrics(test_matrix[test_inds], scores[test_inds], train_matrix[test_inds])
        print("Test:")
        print(metr_test)
        metr_valid = metrics(val_matrix[val_inds], scores[val_inds], train_matrix[val_inds])
        print("Valid:")
        print(metr_valid)


#SGD-based

optimizer = torch.optim.AdamW(lightGCN.parameters(), lr = 0.0005)

mas = []
ind = 0
lightGCN = lightGCN.to(device)
vec_change = []
lightGCN.E0.weight = lightGCN.E0.weight.requires_grad_(False)
for us in tqdm(val_inds[:200]):
    us = int(us)
    nn.init.xavier_uniform_(lightGCN.En.weight)
    #lightGCN.En.weight.to(device)
    a = torch.linalg.norm(lightGCN.En.weight).item()
    lightGCN.E0.weight.requires_grad_(False)
    for epoch in range(1):
        optimizer.zero_grad()

        # users, pos_items, neg_items = dataload.one_user(ind)
        users, pos_items, neg_items = data_loader(warm_val_df, 1, user_num, item_num)


        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = \
             lightGCN.forward(users, pos_items, neg_items, add = True)


        mf_loss, reg_loss = bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
        #mf_loss, reg_loss = mse_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0)
        reg_loss = DECAY * reg_loss
        final_loss = mf_loss + reg_loss + 500 * torch.linalg.norm(lightGCN.En.weight - torch.mean(lightGCN.E0.weight))
        #lightGCN.E0.weight[group_new_users] = lightGCN.En.weight[group_new_users]

        final_loss.backward()
        optimizer.step()
    b = torch.linalg.norm(lightGCN.En.weight).item()
    ind += 1
    if torch.isnan(torch.tensor(abs(b - a))):
        mas += [lightGCN.E0.weight[us].clone()]
        vec_change += [0]
    else:
        mas += [lightGCN.En.weight.clone()]
        vec_change += [abs(b - a)]
        

print(np.mean(vec_change))
        
print(torch.linalg.norm(E - lightGCN.E0.weight))       

with torch.no_grad():
    for ind, us in tqdm(enumerate(val_inds[:200])):
        lightGCN.E0.weight[us] = mas[ind]




lightGCN.eval()
with torch.no_grad():
    
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers()
        final_user_Embed = final_user_Embed.cpu()
        final_item_Embed = final_item_Embed.cpu()
        initial_user_Embed = initial_user_Embed.cpu()
        initial_item_Embed = initial_item_Embed.cpu()
        
        scores = torch.matmul(final_user_Embed, torch.transpose(final_item_Embed,0, 1))
        metr_test = metrics(test_matrix[test_inds], scores[test_inds], train_matrix[test_inds])
        print("Test:")
        print(metr_test)
        metr_valid = metrics(val_matrix[val_inds], scores[val_inds], train_matrix[val_inds])
        print("Valid:")
        print(metr_valid)

#ADMM-based

from scipy.optimize import minimize
# E = torch.load("/kaggle/input/lightgcn-weights-ml1m/embedding_tensor (2).pt")
E = torch.load("LightGCN_weights_clothes.pt")
lightGCN = LightGCN(train_df, user_num, item_num, n_layers, latent_dim)
lightGCN.E0.weight = nn.Parameter(E)
lightGCN.to(device)


final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed, laplas_user, laplas_item = lightGCN.propagate_through_layers_new(pd.concat([warm_val_df, train_df]), val_inds)
mas = []
train_new_matrix = warm_val_matrix 
ro = 0.1
alph = 100
for us in tqdm(val_inds[:100]):
    a0 = train_new_matrix[us].numpy()
    a1 = torch.matmul(laplas_user[us, :-1].unsqueeze(0), torch.cat((initial_user_Embed.cpu(), initial_item_Embed.cpu()), 0)[:-1])
    A0 = final_item_Embed.cpu() - torch.matmul(laplas_item[:, us].unsqueeze(1) / laplas_user[us, us], a1)
    b0 = laplas_item[:, us].unsqueeze(1) / laplas_user[us, us]
    A0 = A0.detach().numpy()
    b0 = b0.detach().numpy()
    random_numbers = random.sample(range(0, final_user_Embed.cpu().shape[0]), 5)
    v_m = torch.mean(initial_user_Embed[random_numbers].cpu(), 0).detach().numpy()
    v_2 = initial_user_Embed[us].cpu().detach().numpy()
    v_m = v_2 * 0.99 + v_m * 0.01
    a1 = a1.cpu().detach().numpy()
    d = laplas_user[us, us].cpu().detach().numpy() 

    I = np.diag(np.ones(A0.shape[1]))

    M = np.linalg.inv(2 * alph * d * I + ro * I - A0.T @ A0 - 4 * b0.T @ a0 * I)
    x = v_2
    z = v_2
    w = v_2
    for _ in range(5):
        x = M @ (2 * alph * (v_m * d) + 2 * a0.T @ A0 + ro * (z - w))
        z = (x + w) / 1000
        w = w + x - z
    
        
        


    v_opt = x
    mas += [(torch.tensor(v_opt - a1) / d)]

with torch.no_grad():
    for ind, us in tqdm(enumerate(val_inds[:100])):
        lightGCN.E0.weight[us] = mas[ind] * 0.1 + lightGCN.E0.weight[us].cpu() * 0.1




lightGCN.eval()
with torch.no_grad():
    
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers()
        final_user_Embed = final_user_Embed.cpu()
        final_item_Embed = final_item_Embed.cpu()
        initial_user_Embed = initial_user_Embed.cpu()
        initial_item_Embed = initial_item_Embed.cpu()
        
        scores = torch.matmul(final_user_Embed, torch.transpose(final_item_Embed,0, 1))
        metr_test = metrics(test_matrix[test_inds], scores[test_inds], train_matrix[test_inds])
        print("Test:")
        print(metr_test)
        metr_valid = metrics(val_matrix[val_inds], scores[val_inds], train_matrix[val_inds])
        print("Valid:")
        print(metr_valid)
