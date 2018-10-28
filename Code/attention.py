# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from sklearn import metrics
from sklearn import preprocessing
import EER

model = torch.nn.Sequential(
    torch.nn.Linear(512, 2)
).cuda()

softmax = nn.Softmax()
fc = nn.Linear(512, 2).cuda()
norn = nn.BatchNorm2d(512).cuda()
def main():

    global N_Train, N_Test, feat_num, feat_len, dtype
    dtype = torch.FloatTensor
    learning_rate = 1

    CASIA_test_label_ALL = np.load('./Feat/CASIA/label_ALLF_Test.npy')
    CASIA_train_label_ALL = np.load('./Feat/CASIA/label_ALLF_Train.npy')
    CASIA_test_MSR = np.load('./Feat/CASIA/feat_Test_CASIA_MSR.npy')
    CASIA_train_MSR = np.load('./Feat/CASIA/feat_Train_CASIA_MSR.npy')
    CASIA_test_ALL = np.load('./Feat/CASIA/feat_Test_CASIA_ALL.npy')
    CASIA_train_ALL = np.load('./Feat/CASIA/feat_Train_CASIA_ALL.npy')
    #CASIA_test_SSR = np.load('./Feat/CASIA/feat_Test_CASIA_SSR.npy')
    #CASIA_train_SSR = np.load('./Feat/CASIA/feat_Train_CASIA_SSR.npy')

    CASIA_Train = []
    CASIA_Train.append(CASIA_train_ALL)
    CASIA_Train.append(CASIA_train_ALL)
    #CASIA_Train.append(CASIA_train_SSR)


    CASIA_Test = []
    CASIA_Test.append(CASIA_test_ALL)
    CASIA_Test.append(CASIA_test_ALL)
    #CASIA_Test.append(CASIA_test_SSR)
    #print np.array(CASIA_Train).shape

    ##define the global param for reshape
    feat_num = np.array(CASIA_Train).shape[0]
    N_Train = np.array(CASIA_Train).shape[1]
    N_Test = np.array(CASIA_Test).shape[1]
    feat_len = np.array(CASIA_Train).shape[2]
    ## convert the numpy to tensor
    CASIA_Train = torch.from_numpy(np.array(CASIA_Train))
    CASIA_Test = torch.from_numpy(np.array(CASIA_Test))

    CASIA_train_label_ALL = torch.from_numpy(np.array(CASIA_train_label_ALL,dtype=int))
    CASIA_test_label_ALL = torch.from_numpy(np.array(CASIA_test_label_ALL,dtype=int))
    CASIA_Train = torch.autograd.Variable(CASIA_Train.type(dtype).cuda())
    CASIA_Test = torch.autograd.Variable(CASIA_Test.type(dtype).cuda())
    ##define the kernel param of the same size with the single feature 
    q = Variable(torch.ones(512, 1).type(dtype).cuda(),requires_grad=True)

    # define loss function (criterion) and optimizer
    # kernel and model use the different optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    opt_Momentum    = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8,weight_decay=0.8)
    opt_Momentum2    = torch.optim.SGD({q}, lr=0.1, momentum=0.8,weight_decay=0.8)
    # reshape the feature size from (feat_num, N, feat_len) to (N*feat_num,feat_len)
    CASIA_Train_reshape = CASIA_Train.view([N_Train*feat_num,feat_len])
    CASIA_Test_reshape = CASIA_Test.view([N_Test*feat_num,feat_len])
    #change the channel order for convolution, shape change from  (feat_num, N, feat_len) to (N,feat_num,feat_len) 
    CASIA_Train_input = CASIA_Train.permute(1,0,2)
    CASIA_Test_input = CASIA_Test.permute(1,0,2)

    for i in range(100):
        train(CASIA_Train_reshape, CASIA_Train_input, CASIA_train_label_ALL, model, criterion, opt_Momentum, opt_Momentum2,q)
        val(CASIA_Test_reshape, CASIA_Test_input, CASIA_test_label_ALL, criterion, model, q)



def train(train_data, train_input, train_label, model, criterion, optimizer, optimizer2, q):
    target_var = torch.autograd.Variable(train_label.cuda())
    # define fusion feature
    fusion = Variable(torch.Tensor(N_Train, feat_len).type(dtype).cuda(), requires_grad=False)
    # dot produce (N, feat_num, feat_len)*(feat_len, 1) 
    temp = torch.mm(train_data,q)
    temp = temp.view([N_Train,feat_num])
    # Normalize the weight 
    temp = softmax(temp)
    # reshape the weight
    temp = temp.view([N_Train,1,feat_num])
    # fuse the feature via weighted average for each pair of features
    for i in range(N_Train):
	fusion[i,:] = torch.mm(temp[i,:],train_input[i,:])
    #print fusion.size()
    #fusion = norn(fusion)
    #fusion.data = torch.norm(fusion.data, p=2)
    #print fusion.size()

    # fc for classification
    predit = model(fusion)
    predit = softmax(predit)
    _, pre = torch.max(predit.data, 1)
    all_ac_score = metrics.accuracy_score(train_label.cpu().numpy(), pre.cpu().numpy())
    print('Accuracy of the network on the train images is')
    print(all_ac_score)
    
    loss = criterion(predit, target_var)
    optimizer.zero_grad()
    optimizer2.zero_grad()
    loss.backward(retain_graph=True)

    print loss.data[0]
    
    optimizer.step()
    optimizer2.step()
    np.save('./Feat/REPLAY/feat_Train_REPLAY_fusion.npy',fusion.cpu().detach().numpy())

def val(val_data, val_input, val_label, criterion, model, q):
    target_var = torch.autograd.Variable(val_label.cuda())
    fusion = Variable(torch.Tensor(N_Test, feat_len).type(dtype).cuda(), requires_grad=False)
    temp = torch.mm(val_data,q)
    temp = temp.view([N_Test,feat_num])

    temp = softmax(temp)
    temp = temp.view([N_Test,1,feat_num])

    for i in range(N_Test):
	fusion[i,:] = torch.mm(temp[i,:],val_input[i,:])
    #fusion = norn(fusion)
    #fusion.data = torch.norm(fusion.data, p=2)
    #print fusion.size()
    predit = model(fusion)
    predit = softmax(predit)
    _, pre = torch.max(predit.data, 1)
    all_ac_score = metrics.accuracy_score(val_label.cpu().numpy(), pre.cpu().numpy())
    print('Accuracy of the network on the test images is')
    print(all_ac_score)
    auc, eer,thd = EER.EER_new(target_var, predit.data.cpu().numpy()[:,1], 'feat1')
    h = EER.HTER_NEW(target_var, predit.data.cpu().numpy()[:,1], 0.5 ,'feat1')
    loss = criterion(predit, target_var) 
    print loss.data[0]
 
    np.save('./Feat/CASIA/feat_Test_CASIA_fusion.npy',fusion.cpu().detach().numpy())
if __name__ == '__main__':
    main()
