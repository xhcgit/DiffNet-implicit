# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import time
from model import Test
import argparse
import time
from process import loadData
from Interface.BPRData import BPRData
import Interface.evaluate as evaluate
import torch.utils.data as dataloader

modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")


class DIFFNET():
    def getData(self, args):
        data = loadData(args.dataset, args.cv)
        if args.dataset == "Tianchi_time":
            interatctMat, testData, trustMat = data
            trainMat = interatctMat
        else:
            trainMat, testData, trustMat = data

        return trainMat, testData, trustMat

    def __init__(self, args):#, train, trust, train_loader, test_loader):
        self.args = args
        trainMat, test_data, trustMat = self.getData(args)
        trainMat = (trainMat != 0) * 1
        train_u, train_v = trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1),train_v.reshape(-1,1))).tolist()#//(u,v)list
        self.test_data = test_data

        self.trustMat = trustMat
        self.trainMat = trainMat
        self.userNum, self.itemNum = self.trainMat.shape

        self.train_dataset = BPRData(train_data, self.itemNum, trainMat, 1, True) 
        self.test_dataset = BPRData(test_data, self.itemNum, trainMat, 0, False)
        self.train_loader = dataloader.DataLoader(self.train_dataset, batch_size=args.batch, shuffle=True, num_workers=0) #//dataloader
        self.test_loader = dataloader.DataLoader(self.test_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)

        self.hide_dim = self.args.hide_dim
        self.r_weight = self.args.reg
        # self.loss_rmse = nn.MSELoss(reduction='sum')#不求平均
        
        self.curEpoch = 0
        #历史数据，用于画图
        self.train_losses = []
        self.train_RMSEs  = []
        self.train_MAEs   = []
        self.test_losses  = []
        self.test_RMSEs   = []
        self.test_MAEs    = []
        self.step_rmse    = []
        self.step_mae     = []

    #初始化参数
    def prepareModel(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)

        self.model = Test(self.userNum, self.itemNum, self.hide_dim).cuda()
        self.model.buildMat(self.trainMat, self.trustMat)
        self.opt = t.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)


    def run(self):
        #判断是导入模型还是重新训练模型
        self.prepareModel()
        Wait = 0
        best_hr = -1
        best_ndcg = -1
        for e in range(self.curEpoch, self.args.epochs+1):
            #记录当前epoch,用于保存Model
            self.curEpoch = e
            #分别使用两个autoencoder训练embedding
            log("**************************************************************")
            #训练
            log("begin train")
            epoch_loss = self.trainModel()
            log("end train")
            log("epoch %d/%d, epoch_loss=%.4f"%(e,self.args.epochs, epoch_loss))

            #测试
            HR,NDCG = self.testModel()
            log("HR=%.4f, NDCG=%.4f"%(HR,NDCG))

            if HR >= best_hr:
                best_hr = HR
                best_ndcg = NDCG
                Wait = 0
                best_epoch = self.curEpoch
            else:
                Wait += 1
                log("Wait = %d"%(Wait))
            if Wait >= 5:
                HR,NDCG = self.testModel(save=True)
                uids = np.array(self.test_data[::101])[:,0]
                data = {}
                assert len(uids) == len(HR)
                assert len(uids) == len(np.unique(uids))
                for i in range(len(uids)):
                    uid = uids[i]
                    data[uid] = [HR[i], NDCG[i]]

                with open("DiffNet-{0}-cv{1}-test.pkl".format(self.args.dataset, self.args.cv), 'wb') as fs:
                    pickle.dump(data, fs)

                break

        log('best epoch = %d, best hr = %.4f, best ndcg = %.4f'%(best_epoch, best_hr, best_ndcg))

        name = self.getModelName()
        log("model name : %s"%(name))

    

    def trainModel(self):
        epoch_loss = 0
        self.train_loader.dataset.ng_sample()
        for user, item_i, item_j in self.train_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()

            pred_pos = self.model(user, item_i)
            pred_neg = self.model(user,item_j)
            bprloss = -(pred_pos.view(-1) - pred_neg.view(-1)).sigmoid().log().sum()
            
            epoch_loss += bprloss.item()
            loss = bprloss/self.args.batch

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return epoch_loss

    def sparseTest(self, trainMat, testMat):
        interationSum = np.sum(trainMat != 0)
        flag = int(interationSum/3)
        user_interation = np.sum(trainMat != 0, axis=1).reshape(-1).A[0]
        sort_idx = np.argsort(user_interation)
        user_interation_sort = user_interation[sort_idx]
        
        tmp = 0
        idx = []
        for i in range(user_interation_sort.size):
            if tmp >= flag:
                tmp = 0
                idx.append(i)
                continue
            else:
                tmp += user_interation_sort[i]
        print("<{0}, <{1}, <{2}".format(user_interation_sort[idx[0]], \
                                            user_interation_sort[idx[1]], \
                                            user_interation_sort[-1]))
        print("{0}, {1}, {2}".format(idx[0], idx[1]-idx[0], self.userNum-idx[1]))
        splitUserIdx = [sort_idx[0:idx[0]], sort_idx[idx[0]: idx[1]], sort_idx[idx[1]:]]
        self.sparseTestModel(sort_idx)
        for i in splitUserIdx:
            self.sparseTestModel(i)

    def sparseTestModel(self, uid):
        test_u = np.array(uid[self.testMat[uid].tocoo().row])
        test_v = self.testMat[uid].tocoo().col
        test_r = self.testMat[uid].tocoo().data
        _, rmse, mae = self.testModel(self.testMat[uid], (test_u, test_v, test_r))
        print("sparse test : user num = %d, rmse = %.4f, mae = %.4f"%(uid.size, rmse, mae))


    def testModel(self, save=False):
        HR=[]
        NDCG=[]
        for test_u, test_i in self.test_loader:
            test_u = test_u.long().cuda()
            test_i = test_i.long().cuda()

            pred = self.model(test_u, test_i)
            batch = int(test_u.cpu().numpy().size/101)
            for i in range(batch):
                batch_socres=pred[i*101:(i+1)*101].view(-1)
                _,indices=t.topk(batch_socres,args.topk) #根据分数找出topk的索引
                tmp_item_i=test_i[i*101:(i+1)*101]
                recommends=t.take(tmp_item_i,indices).cpu().numpy().tolist()
                gt_item=tmp_item_i[0].item()
                HR.append(evaluate.hit(gt_item,recommends))
                NDCG.append(evaluate.ndcg(gt_item,recommends))
        if save:
            return HR, NDCG
        else:
            return np.mean(HR),np.mean(NDCG)


    def getModelName(self):
        title = "DiffNet_"
        ModelName = title + dataset + "_" + modelUTCStr + \
        "_reg_" + str(self.r_weight)+ \
        "_hide_" + str(self.hide_dim) + \
        "_batch_" + str(self.args.batch) +\
        "_lr_" + str(self.args.lr)
        return ModelName


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR-GMI main.py')
    parser.add_argument('--dataset', type=str, default="Tianchi")
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--seed', type=int, default=29, metavar='int', help='random seed')

    parser.add_argument('--hide_dim', type=int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--reg', type=float, default=0, metavar='N', help='reg weight')
    parser.add_argument('--batch', type=int, default=4096, metavar='N', help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')

    parser.add_argument('--test_batch', type=int, default=2048, metavar='N', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train')
    parser.add_argument('--patience', type=int, default=5, metavar='int', help='early stop patience')
    parser.add_argument('--topk',type=int, default=10)

    args = parser.parse_args()
    args.dataset += "_time"
    dataset = args.dataset


    hope = DIFFNET(args)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()

