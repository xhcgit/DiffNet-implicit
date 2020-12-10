import torch as t
import torch.nn as nn
import numpy as np

class Test(nn.Module):
    def __init__(self, userNum, itemNum, hide_dim):
        super(Test, self).__init__()
        self.userEmbed = nn.Embedding(userNum, hide_dim)
        self.itemEmbed = nn.Embedding(itemNum, hide_dim)

        # self.predLayer = nn.Sequential(
        #     nn.Linear(hide_dim*2, hide_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(hide_dim*2, 1)
        # )
        nn.init.xavier_normal_(self.userEmbed.weight)
        nn.init.xavier_normal_(self.itemEmbed.weight)

    def buildMat(self, trainMat, trustMat):
        self.trainMat = trainMat.tocoo()
        self.trustMat = trustMat.tocoo()

        train_indices = t.from_numpy(
            np.vstack((self.trainMat.row, self.trainMat.col)).astype(np.int64))
        train_shape = t.Size(trainMat.shape)

        trust_indices = t.from_numpy(
            np.vstack((self.trustMat.row, self.trustMat.col)).astype(np.int64))
        trust_shape = t.Size(trustMat.shape)

        uid = self.trainMat.row
        iid = self.trainMat.col
        ratingNum = np.sum(trainMat!=0, axis=1).A.reshape(-1)
        assert np.sum(ratingNum == 0) == 0
        norm_train = np.zeros_like(self.trainMat.data, dtype=np.float).reshape(-1)
        for i in range(norm_train.size):
            norm_train[i] = 1/ratingNum[uid[i]]  #除以user 打分个数
        norm_train = t.from_numpy(norm_train).view(-1)

        self.consumed_items_sparse_matrix = t.sparse.FloatTensor(train_indices, norm_train, train_shape).float().cuda()


        uid_trust = self.trustMat.row
        uid_trustee = self.trustMat.col
        trustNum = np.sum(trustMat != 0, axis=1).A.reshape(-1)
        assert np.sum(trustNum == 0) == 0
        norm_trust = np.zeros_like(self.trustMat.data, dtype=np.float).reshape(-1)
        for i in range(norm_trust.size):
            norm_trust[i] = 1/trustNum[uid_trust[i]]
        norm_trust = t.from_numpy(norm_trust).view(-1)
        self.social_neighbors_sparse_matrix = t.sparse.FloatTensor(trust_indices, norm_trust, trust_shape).float().cuda()


    def forward(self, user_idx, item_idx):
        user_embedding_from_consumed_items = t.spmm(self.consumed_items_sparse_matrix, self.itemEmbed.weight)

        first_gcn_user_embedding = t.spmm(self.social_neighbors_sparse_matrix, self.userEmbed.weight)
        second_gcn_user_embedding = t.spmm(self.social_neighbors_sparse_matrix, first_gcn_user_embedding)

        self.final_user_embedding = second_gcn_user_embedding + user_embedding_from_consumed_items

        latest_user_latent = self.final_user_embedding[user_idx]
        latest_item_latent = self.itemEmbed(item_idx)

        # tensor = t.cat((latest_user_latent, latest_item_latent), dim=1)
        # prediction = self.predLayer(tensor)

        predict_vector = t.mul(latest_user_latent, latest_item_latent)
        prediction = t.sum(predict_vector, 1, keepdims=True)
        return prediction

    # def weights_init(self):
    #     if self.weight is not None:
    #         nn.init.xavier_uniform_(self.weight)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)




