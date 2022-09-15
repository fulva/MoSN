import torch as th
import numpy as np
from aug import augmentation
from PIL import Image, ImageFilter, ImageOps

th.set_printoptions(precision = 2, sci_mode = False)

def get_single(state_data, device = None):
    label = []
    data = []
    aug_trans = augmentation("single")
    for i in range(len(state_data)):
        label.append(np.array(state_data[i][0]))
        data.append(np.array(aug_trans(Image.fromarray(state_data[i][1])))/255)
    label = th.FloatTensor(label).float()
    data = th.FloatTensor(np.array(data)).float()
    label = label.to(device)
    data = data.contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device)
    return label, data

def get_pairs(data, aug_type):
    if aug_type == "train":
        label1 = []
        label2 = []
        data1 = []
        data2 = []
        aug_trans = augmentation(aug_type)
        while len(data1) < 10000:
    	    for i in range(len(data)):
                label1.append(np.array(data[i][0]))
                label2.append(np.array(data[i][0]))
                im1, im2 = aug_trans(data[i][1], data[i][1])
                data1.append(np.array(im1)/255)
                data2.append(np.array(im2)/255)
        label1 = th.FloatTensor(np.array(label1)).float()
        label2 = th.FloatTensor(np.array(label2)).float()
        data1 = th.FloatTensor(np.array(data1)).float()
        data2 = th.FloatTensor(np.array(data2)).float()
        return label1, label2, data1, data2
    if aug_type == "test":
        label1 = []
        data1 = []
        for i in range(len(data)):
            label1.append(np.array(data[i][0]))
            data1.append(np.array(data[i][1])/255)
        label1 = th.FloatTensor(np.array(label1)).float()
        data1 = th.FloatTensor(np.array(data1)).float()
        return label1, data1

def for_test(label1, data1, label2 = None, data2 = None, device = None):
    if len(data1) > 10000:
        idx1 = th.randperm(len(data1))[:10000]
        idx2 = th.randperm(len(data1))[:10000]
    else:
        idx1 = th.randperm(10000)%len(data1)
        idx2 = th.randperm(10000)%len(data1)
    xlabel1 = label1[idx1].to(device)
    if label2 == None:
        xlabel2 = label1[idx2].to(device)
    else:
        xlabel2 = label2[idx2].to(device)
    xdata1 = data1[idx1].contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device) 
    if data2 == None:
        xdata2 = data1[idx2].contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device)
    else:
        xdata2 = data2[idx2].contiguous().view(-1, 32, 32, 3).permute(0,3,1,2).to(device)
    
    return xlabel1, xlabel2, xdata1, xdata2

def error(F_b, F_m, data1, label, method, data2 = None):
    num = len(label[0])

    if (len(data1)==0):
        error = "None"
    else:
        if data2 != None:
            pred = (F_b(data2)+F_m(data2))/2 - (F_b(data1)+F_m(data1))/2
        else:
            pred = (F_b(data1)+F_m(data1))/2
        if method == "mae":
            error = (th.abs(pred - label)).sum(1).mean()/num
            error = error.cpu().detach().numpy()
        elif method == "mde":
            error = (th.abs(th.round(pred) - label)).sum(1).mean()/num
            error = error.cpu().detach().numpy()
        else:
            print("    The method is not included, please check.")
    return error

def print_error(F_b, F_m, state_fortest, train_fortest, test_fortest, method):

    state_label, state_data = state_fortest
    train_label1, train_label2, train_data1, train_data2 = train_fortest
    test_label1, test_label2, test_data1, test_data2 = test_fortest

    state_error = error(F_b, F_m, state_data,  state_label, method)
    train_error = error(F_b, F_m, train_data1, train_label2 - train_label1, method, train_data2)
    test_error = error(F_b, F_m, test_data1, test_label2 - test_label1, method, test_data2)

    if state_error != "None":
        print("    state_" +method +": " + str(state_error), end='')
    if train_error != "None":
        print("    train_"+method+": " +str(train_error), end='')
    if test_error != "None":
        print("    test_" +method +": " + str(test_error), end='')

    return state_error, train_error, test_error

def psr(F_b, F_m, label1, label2, data1, data2):
    pred = (F_b(data2) + F_m(data2))/2 - (F_b(data1) + F_m(data1))/2
    psr =  (th.abs(th.round(pred) - (label2-label1))).sum(1).gt(0).sum()/len(pred)
    psr = 1 - psr.cpu().detach().numpy()
    return psr

def print_psr(F_b, F_m, test_fortest):
    test_label1, test_label2, test_data1, test_data2 = test_fortest
    test_psr = psr(F_b, F_m, test_label1, test_label2, test_data1, test_data2)

    if test_psr!= "None":
        print("    psr" +": " + str(test_psr), end='')

    return test_psr


