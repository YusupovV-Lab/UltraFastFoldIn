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
    #print(targets.shape)
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

def data_loader(data, batch_size, n_usr, n_itm):
  
    interected_items_df = data.groupby('user_id')['item_id'].apply(list).reset_index()
    
  
    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id
  
    indices = list(data['user_id'].unique())
    
    
    if len(indices) < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)

    users.sort()
  
    users_df = pd.DataFrame(users,columns = ['users'])

    interected_items_df = pd.merge(interected_items_df, users_df, how = 'right', left_on = 'user_id', right_on = 'users')
  
    pos_items = interected_items_df['item_id'].apply(lambda x : random.choice(list(x))).values

    neg_items = interected_items_df['item_id'].apply(lambda x: sample_neg(list(x))).values

    return list(users), list(pos_items), list(neg_items)




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

loss_list_epoch = []
MF_loss_list_epoch = []
reg_loss_list_epoch = []

recall_list = []
precision_list = []
ndcg_list = []
map_list = []
hr_list = []
mrr_list = []

#E_p = torch.load("/kaggle/input/lightgcn-weights-ml1m/embedding_tensor (2).pt")



train_time_list = []
eval_time_list = [] 
#lightGCN.E0.weight = nn.Parameter(E, requires_grad = True)
best_hr = -1
for epoch in range(EPOCHS):
    print(epoch)
    n_batch = int(len(train)/(BATCH_SIZE))
  
    final_loss_list = []
    MF_loss_list = []
    reg_loss_list = []
  
    best_ndcg = -1
  
    train_start_time = time.time()
    lightGCN = lightGCN.to(device)
    lightGCN.train()
    for batch_idx in tqdm(range(n_batch)):

        optimizer.zero_grad()

        users, pos_items, neg_items = data_loader(train_df, BATCH_SIZE, user_num, item_num)

        users_emb, item_emb, userEmb0, itemEmb0 = lightGCN.forward_mse(users)

        mf_loss, reg_loss = mse_loss2(users_emb, userEmb0, item_emb, itemEmb0, train_matrix[users].to(device))
        reg_loss = DECAY * reg_loss
        final_loss = mf_loss + reg_loss #+ 0.001 * mf_loss_mse

        final_loss.backward()
        optimizer.step()
        #scheduler.step()

        final_loss_list.append(final_loss.item())
        MF_loss_list.append(mf_loss.item())
        reg_loss_list.append(reg_loss.item())


    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    # lightGCN.E0.weight = nn.Parameter((1 / 2 + epoch / 10) * lightGCN.E0.weight + (1 / 2 - epoch / 10) * E_p)

    print(lightGCN.E0.weight.norm())

    lightGCN.eval()
    with torch.no_grad():
    
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers()
        final_user_Embed = final_user_Embed.cpu()
        final_item_Embed = final_item_Embed.cpu()
        initial_user_Embed = initial_user_Embed.cpu()
        initial_item_Embed = initial_item_Embed.cpu()
        
        scores = torch.matmul(final_user_Embed, torch.transpose(final_item_Embed,0, 1))
        metr_val = metrics(val_matrix[val_inds], scores[val_inds], train_matrix[val_inds])
        print("Val:")
        print(metr_val)
        if metr_val['hits@10'].item() > best_hr:
            E = lightGCN.E0.weight.data.clone()
            best_hr = metr_val['hits@10'].item()
            print("Upd")
        metr_test = metrics(test_matrix[test_inds], scores[test_inds], train_matrix[test_inds])
        print("Test:")
        print(metr_test)
        
        
        
     

    eval_time = time.time() - train_end_time

    loss_list_epoch.append(round(np.mean(final_loss_list),4))
    MF_loss_list_epoch.append(round(np.mean(MF_loss_list),4))
    reg_loss_list_epoch.append(round(np.mean(reg_loss_list),4))



    train_time_list.append(train_time)
    eval_time_list.append(eval_time)

torch.save(E, "LightGCN_weights_clothes.pt")
