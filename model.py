import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm


class LACF(nn.Module):
    def __init__(self, data_config, args):
        super(LACF, self).__init__()
        print('LACFFinal')
        self.device = args.device
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']  
        self.all_t_list = data_config['all_t_list']  
        self.A_in_shape = self.plain_adj.tocoo().shape  # (u+i,u+i)
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).to(self.device) 
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).to(self.device)  
        self.all_h_list = torch.LongTensor(self.all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(self.all_t_list).to(self.device)
        self.G_indices, self.G_values = self._cal_sparse_adj()
        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers  # 2
        self.n_intents = args.n_intents  # 128
        self.temp = args.temp
        self.lrec = args.lrecon
        self.rt = args.temp
        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg  # 2e-5
        self.cen_reg = args.cen_reg  # 0.005
        self.ssl_reg = args.ssl_reg  # 0.1
        self.auto_t = args.beta
        self.zu, self.zuu, self.zi, self.zii, self.zut, self.zit = [None] * self.n_layers, [None] * self.n_layers, [None] * self.n_layers, [None] * self.n_layers, [None] * self.n_layers, [None] * self.n_layers
        self.hu, self.hi, self.eu, self.ei = None, None, None, None
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)  # (u,d)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)  # (I,d)
        self._init_weight()
        self.input_dim = args.embed_size
        self.mlp_edge_model_dim = self.emb_dim
        self.mlp_edge_model = nn.ModuleList([nn.Sequential(
            nn.Linear(self.input_dim * 2, self.mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_edge_model_dim, 1)
        ) for i in range(self.n_layers)])
        self.mlp_emb_model = nn.ModuleList([nn.Sequential(
            nn.Linear(self.input_dim, self.mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_edge_model_dim, self.mlp_edge_model_dim)
        ) for i in range(self.n_layers)])


    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def neg_sample(self, users, items):
        user_set = {}
        for i, user in enumerate(tqdm(users)):
            if user not in user_set:
                user_set[user] = [items[i]]
            else:
                user_set[user].append(items[i])
        neg = []
        for user in tqdm(user_set):
            while True:
                temp = np.random.randint(0, self.n_items - 1)
                if temp in user_set[user]:
                    continue
                else:
                    neg.append(temp)
                    break
        return torch.tensor(neg).to(self.device)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).to(self.device) 
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values,
                                             sparse_sizes=self.A_in_shape).to(self.device)  # (54335, 54335)
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        self.A_in_shape = self.plain_adj.tocoo().shape  # (u+i,u+i)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values,
                                                  self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0],
                                                  self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def auto_augment_s(self, gnn, i, is_norm=False):
        src = torch.index_select(gnn, 0, self.all_h_list)  # torch.Size([2344850, 32])
        dst = torch.index_select(gnn, 0, self.all_t_list)  # torch.Size([2344850, 32])
        edge_emb = torch.cat([src, dst], 1)  # torch.Size([2344850, 64])
        edge_logits = self.mlp_edge_model[i](edge_emb)  # torch.Size([2344850, 1])
        bias = 0.0001
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)  # torch.Size([2344850, 1])
        # edge_score = torch.log(eps) - torch.log(1 - eps)
        edge_score = -torch.log(-torch.log(eps))
        edge_score = edge_score.to(self.device)
        edge_score = (edge_score + edge_logits) / self.auto_t
        # edge_score = (edge_score * edge_logits)
        # edge_score = edge_logits
        batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze().detach()  # torch.Size([2344850])
        zqg = batch_aug_edge_weight
        self.edge_reg += zqg.sum() / (self.n_users + self.n_items)
        if is_norm:
            A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=zqg,sparse_sizes=self.A_in_shape).to(self.device)
            D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
            G_values = D_scores_inv[self.all_h_list] * zqg
            return G_values
        else:
            return zqg

    def auto_augment_f(self, i, f_emb):
        emb_logits = self.mlp_emb_model[i](f_emb)  # torch.Size([2344850, 1])
        bias = 0.0001
        eps = (bias - (1 - bias)) * torch.rand(emb_logits.size()) + (1 - bias)  # torch.Size([2344850, 1])
        emb_score = -torch.log(-torch.log(eps))
        emb_score = emb_score.to(self.device)
        emb_score = (emb_score + emb_logits) / self.auto_t
        batch_aug_emb = torch.sigmoid(emb_score).squeeze().detach()  # torch.Size([2344850])
        self.node_reg += batch_aug_emb.sum() / (self.n_users + self.n_items)
        return batch_aug_emb * f_emb

    def inference1(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings1 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings2 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        # all_embeddings2 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        for i in range(0, self.n_layers):
            gnn = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1],all_embeddings[i])
            self.zu[i], self.zi[i] = torch.split(gnn, [self.n_users, self.n_items], 0)
            zqg1 = self.auto_augment_s(all_embeddings1[i], i, is_norm=True)
            gnn1 = torch_sparse.spmm(self.G_indices, zqg1, self.A_in_shape[0], self.A_in_shape[1], all_embeddings1[i])
            self.zuu[i], self.zii[i] = torch.split(gnn1, [self.n_users, self.n_items], 0)
            new_e = self.auto_augment_f(i, all_embeddings2[i])
            gnnf = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], new_e)
            self.zut[i], self.zit[i] = torch.split(gnnf, [self.n_users, self.n_items], 0)
            all_embeddings.append(gnn + all_embeddings[i])
            all_embeddings1.append(gnn1 + all_embeddings1[i])
            all_embeddings2.append(gnnf + all_embeddings2[i])
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.eu, self.ei = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        all_embeddings1 = torch.stack(all_embeddings1, dim=1)
        all_embeddings1 = torch.sum(all_embeddings1, dim=1, keepdim=False)
        all_embeddings2 = torch.stack(all_embeddings2, dim=1)
        all_embeddings2 = torch.sum(all_embeddings2, dim=1, keepdim=False)
        self.eu1, self.ei1 = torch.split(all_embeddings1, [self.n_users, self.n_items], 0)
        self.eu2, self.ei2 = torch.split(all_embeddings2, [self.n_users, self.n_items], 0)

    def cal_loss(self, emb1, emb2, t):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / t)  # (8686,32)*(8686,32)->(8686)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / t), axis=1)  # (8686,32)@(32,8686)=(8686,8686)->(8686)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def cse_loss(self, users, items, zu, zuu, zi, zii, t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        for i in range(self.n_layers):
            u1 = F.normalize(zu[i][users], dim=1)
            i1 = F.normalize(zi[i][items], dim=1)
            u2 = F.normalize(zuu[i][users], dim=1)
            i2 = F.normalize(zii[i][items], dim=1)
            cl_loss += self.cal_loss(u1, u2, t)  
            cl_loss += self.cal_loss(i1, i2, t)  
        return cl_loss

    def globalcl(self, users, items, eu, hu, ei, hi, t):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        eu = F.normalize(eu[users], dim=1)
        ei = F.normalize(ei[items], dim=1)
        hu = F.normalize(hu[users], dim=1)
        hi = F.normalize(hi[items], dim=1)
        cl_loss += self.cal_loss(eu, hu, t)  
        cl_loss += self.cal_loss(ei, hi, t)  
        return cl_loss



    def inference_test(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings1 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        all_embeddings2 = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        for i in range(0, self.n_layers):
            gnn = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1],
                                    all_embeddings[i])
            self.zu[i], self.zi[i] = torch.split(gnn, [self.n_users, self.n_items], 0)
            zqg1 = self.auto_augment_s(all_embeddings1[i], i, is_norm=True)
            gnn1 = torch_sparse.spmm(self.G_indices, zqg1, self.A_in_shape[0], self.A_in_shape[1], all_embeddings1[i])
            self.zuu[i], self.zii[i] = torch.split(gnn1, [self.n_users, self.n_items], 0)
            new_e = self.auto_augment_f(i, all_embeddings2[i])
            gnnf = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], new_e)
            self.zut[i], self.zit[i] = torch.split(gnnf, [self.n_users, self.n_items], 0)
            all_embeddings.append(gnn + all_embeddings[i])
            all_embeddings1.append(gnn1 + all_embeddings1[i])
            all_embeddings2.append(gnnf + all_embeddings2[i])
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.eu, self.ei = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        all_embeddings1 = torch.stack(all_embeddings1, dim=1)
        all_embeddings1 = torch.sum(all_embeddings1, dim=1, keepdim=False)
        all_embeddings2 = torch.stack(all_embeddings2, dim=1)
        all_embeddings2 = torch.sum(all_embeddings2, dim=1, keepdim=False)
        self.eu1, self.ei1 = torch.split(all_embeddings1, [self.n_users, self.n_items], 0)
        self.eu2, self.ei2 = torch.split(all_embeddings2, [self.n_users, self.n_items], 0)
        return all_embeddings1, all_embeddings2

    def bpr_loss(self, users, pos, neg, u_emb, i_emb):
        u_embeddings = u_emb[users]
        pos_embeddings = i_emb[pos]  # (1024,32)
        neg_embeddings = i_emb[neg]  # (1024,32)
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return mf_loss

    def forward(self, users, pos_items, neg_items):
        users = torch.LongTensor(users).to(self.device)
        pos_items = torch.LongTensor(pos_items).to(self.device)
        neg_items = torch.LongTensor(neg_items).to(self.device)
        self.inference1()
        bpr1 = self.bpr_loss(users, pos_items, neg_items, self.eu, self.ei)
        mf_loss = bpr1
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        l_loss = self.ssl_reg * self.cse_loss(users, pos_items, self.zuu, self.zut, self.zii, self.zit, self.temp)
        g_loss = self.lrec * self.res_loss(users, pos_items, self.eu, self.eu1, self.ei, self.ei1, self.rt) + \
                   self.lrec * self.res_loss(users, pos_items, self.eu, self.eu2, self.ei, self.ei2, self.rt)
        return mf_loss, g_loss, l_loss, emb_loss

    def ssl_con_loss(self, x, y, temp=1.0):
        x = F.normalize(x)
        y = F.normalize(y)
        mole = torch.exp(torch.sum(x * y, dim=1) / temp)
        deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)
        return -torch.log(mole / (deno + 1e-8) + 1e-8).mean()

    def res_loss(self, user, pos, u, u2, i, i2, t):  
        recon_u, recon_u2 = u[user], u2[user]
        recon_i, recon_i2 = i[pos], i2[pos]
        recon_lossu = self.ssl_con_loss(recon_u, recon_u2, t)  # /pos.shape[0]
        recon_lossi = self.ssl_con_loss(recon_i, recon_i2, t)  # /pos.shape[0]
        recon_loss = recon_lossu + recon_lossi
        return recon_loss

    def predict(self, users):
        u_embeddings = self.eu[torch.LongTensor(users).to(self.device)]
        i_embeddings = self.ei
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings

    def reg_params(self, model):
        reg_loss = 0
        for W in model.parameters():
            reg_loss += W.norm(2).square()
        return reg_loss
