import numpy as np
import torch as th
import random

from utils import get_pairs, for_test, get_single, print_error, print_psr

random.seed(10)
th.manual_seed(10)


def train(F_b, F_m, m,  data, device, scene, num, seen, steps, bs):
    index = int(len(data)*seen/100)
    train_data = data[:index]
    seen = len(train_data)
    test_data = data[index:]

    print("\n    We have " + str(seen)+ " states to train mosn model.")
    print("\n    We have " + str(len(test_data))+ " states to test mosn model.")

    train_label1, train_label2, train_data1, train_data2 = get_pairs(train_data, "train")
    train_fortest = for_test(train_label1, train_data1, train_label2, train_data2, device)

    state_fortest = get_single(test_data, device)

    test_label, test_data = get_pairs(test_data, "test")
    test_fortest = for_test(test_label, test_data, None, None, device)


    optimizer = th.optim.Adam(F_b.parameters(), lr=0.0001)

    for step in range(steps):
        optimizer.zero_grad()
        if bs < len(train_label1): 
            idx1 = th.randperm(len(train_label1))[:bs]
            idx2 = th.randperm(len(train_label1))[:bs]
        else:
            idx1 = th.randperm(bs)%len(train_label1)
            idx2 = th.randperm(bs)%len(train_label1)

        #Train
        label1 = train_label1[idx1].to(device)
        label2 = train_label2[idx2].to(device)
        data1 = train_data1[idx1].contiguous().view(bs, 32, 32, 3).permute(0,3,1,2).to(device)
        data2 = train_data2[idx2].contiguous().view(bs, 32, 32, 3).permute(0,3,1,2).to(device)

        q1 = F_b(data1)
        q2 = F_b(data2)
        k1 = F_m(data1)
        k2 = F_m(data2)

        loss = (((k2 - q1) - (label2 - label1))**2).sum(1).mean()/(2*num) + (((q2 + k1) - (label2 + label1))**2).sum(1).mean()/(2*num)

        loss.backward()
        
        optimizer.step()
        for param_b, param_m in zip(F_b.parameters(), F_m.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        #Test
    
        if (step+1)%1000 == 0: 
            print("\n    -----------"+str(step)+": " +str((loss).cpu().detach().numpy()) + "--------------------")
            mae = print_error(F_b, F_m, state_fortest, train_fortest, test_fortest,"mae")
            psr = print_psr(F_b, F_m, test_fortest)
