#UltraGCN

class UltraGCN(nn.Module):
    def __init__(self, data, n_users, n_items, n_layers = 1, latent_dim = 32):
        """
        The UltraGCN Recommendation model
        data - the user-item interaction data
        n_users - number of users
        n_items - number of items
        n_layers - number of layers 
        latent_dim - the dimension of user and item vector representations
        """
        super(UltraGCN, self).__init__()
        self.data = data
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.init_embedding()
        u_d, i_d = self.get_A_tilda()
        self.u_d = u_d.to(device)
        self.i_d = i_d.to(device)

    def init_embedding(self):
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.E0.weight)
        #self.E0.weight = nn.Parameter(self.E0.weight)
        
        self.En = nn.Embedding(self.n_users, self.latent_dim)
        nn.init.xavier_uniform_(self.En.weight)
        


    def get_A_tilda(self, full_data = None):
        R = sp.dok_matrix((self.n_users, self.n_items), dtype = np.float32)
        if full_data is None:
            R[self.data['user_id'], self.data['item_id']] = 1.0
        else:
            R[full_data['user_id'], full_data['item_id']] = 1.0

        d_user = np.sum(R, axis=1)
        d_item = np.sum(R, axis=0).reshape(-1)
        b_user = np.sqrt(d_user + 1) / (d_user + 0.00001)
        b_item = 1 / np.sqrt(1 + d_item)

        return torch.tensor(d_user), torch.tensor(d_item)

    
    def propagate_through_layers(self):
        all_layer_embedding = [self.E0.weight]
        E_lyr = self.E0.weight

        initial_user_Embed, initial_item_Embed = torch.split(self.E0.weight, [self.n_users, self.n_items])

        #print((1 + 1 / torch.sqrt(1 + self.i_d)).shape)
        #print(initial_item_Embed.shape)
        final_user_Embed = (1 + torch.sqrt(1 + self.u_d) / self.u_d) * initial_user_Embed
        final_item_Embed = (1 + 1 / torch.sqrt(1 + self.i_d)).transpose(0, 1) * initial_item_Embed

        return final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed

    def forward(self, users, pos_items, neg_items):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = self.propagate_through_layers()

        users_emb, pos_emb, neg_emb = final_user_Embed[users], final_item_Embed[pos_items], final_item_Embed[neg_items]
        userEmb0,  posEmb0, negEmb0 = initial_user_Embed[users], initial_item_Embed[pos_items], initial_item_Embed[neg_items]

        return users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0

    def forward_mse(self, users):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = self.propagate_through_layers()
        users_emb = final_user_Embed[users]
        userEmb0 = initial_user_Embed[users]
        return users_emb, final_item_Embed, userEmb0, initial_item_Embed
    
    
    def propagate_through_layers_new(self, full_data, new_inds, matr = None):
        all_layer_embedding = [torch.cat((self.En.weight, self.E0.weight[self.n_users:]), 0)]
        E_lyr = torch.cat((self.En.weight, self.E0.weight[self.n_users:]), 0)
        
        initial_user_Embed, initial_item_Embed = torch.split(E_lyr, [self.n_users, self.n_items])
        final_user_Embed = (torch.sqrt(1 + self.u_d) / self.u_d) * initial_user_Embed
        final_item_Embed = (1 / torch.sqrt(1 + self.i_d)).transpose(0, 1) * initial_item_Embed

        return final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed

    def forward_new(self, users, pos_items, neg_items, train_df, group_new_users):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = lightGCN.propagate_through_layers_new(train_df, group_new_users)

        users_emb, pos_emb, neg_emb = final_user_Embed[users], final_item_Embed[pos_items], final_item_Embed[neg_items]
        userEmb0,  posEmb0, negEmb0 = initial_user_Embed[users], initial_item_Embed[pos_items], initial_item_Embed[neg_items]

        return users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
        
    def forward_mse_new(self, users, train_df, group_new_users):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed, laplas_user, laplas_item = lightGCN.propagate_through_layers_new(train_df, group_new_users)
        users_emb = final_user_Embed[users]
        userEmb0 = initial_user_Embed[users]
        return users_emb, final_item_Embed, userEmb0, initial_item_Embed


#Losses

#BPR Loss
def bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0):
    """
    BPR Loss
    users - list of users' indices
    user_emb - embeddings of the users
    pos_emb - embeddings of positive items in the sample
    neg_emb - embeddings of negative items in the sample
    userEmb0 - the initial embeddings of the users before graph matrix multiplication 
    posEmb0 - the initial embeddings of positive items in the sample before graph matrix multiplication 
    negEmb0 - the initial embeddings of negative items in the sample before graph matrix multiplication
    """
  
    reg_loss = (1/2)*(userEmb0.norm().pow(2) + 
                    posEmb0.norm().pow(2)  +
                    negEmb0.norm().pow(2))/float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
        
    loss = -torch.mean(torch.log(torch.nn.functional.sigmoid(-neg_scores + pos_scores)))
        
    return loss, reg_loss

#MSE Loss
def mse_loss2(users_emb, userEmb0, item_emb, itemEmb0, train_old):
    """
    MSE Loss
    user_emb - embeddings of the users
    item_emb - embeddings of the items 
    userEmb0 - the initial embeddings of the users before graph matrix multiplication 
    itemEmb0 - the initial embeddings of items in the sample before graph matrix multiplication 
    train_old - ground truth scores
    """
    reg_loss = userEmb0.norm().pow(2) + itemEmb0.norm().pow(2)
    scores = torch.matmul(users_emb, item_emb.transpose(0, 1))
    loss = (train_old - scores).norm().pow(2)
    return loss, reg_loss


    
    
