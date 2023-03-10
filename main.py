import os
import numpy as np
import torch as th
import random
import argparse
from models import initialize_model
from train import train
random.seed(10)

parser = argparse.ArgumentParser(description='MoSN Training')

parser.add_argument('--m', default=0.999, type=float, help='The momentum update factor.')
parser.add_argument('--w-size', default=32, type=int, help='The size of the input image.')
parser.add_argument('--num', default=8, type=int, help='The number of switches.')
parser.add_argument('--seen', default=30, type=int, help='The proportion of different states included in training.')
parser.add_argument('--scene', default='SE-FA', type=str, help='The name of the environment.')


def main():
    args = parser.parse_args()
    print(args)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("----------"+str(device) + " will be used to do the computation"+"----------")

    dirpath = os.path.join(os.path.join(os.getcwd(), "data/"), args.scene)

    if args.scene == "SE-FA":
        data_dict = np.load(os.path.join(dirpath, args.scene+"_all_images_N"+str(args.num)+"_Size"+str(args.w_size)+".npy"), allow_pickle=True).item()
        data = []
        for key, value in data_dict.items():
            data.append((key, value))
        del data_dict
    elif (scene == "DE-FA") | (scene == "SE-DA") | (scene == "DE-DA") :
        data_dict = np.load(os.path.join(dirpath, args.scene+"_all_images_N"+str(args.num)+"_Size"+str(args.w_size)+".npy"), allow_pickle=True).item()
    else:
        print("Scene: "+scene+" is not exist.")

    random.shuffle(data)
    print("    The name of our environment is: "+ str(args.scene))
    print("    The momentum update factor: "+str(args.m))
    print("    The number of switches: "+str(args.num))
    print("    The size of our input image: " + str(data[0][1].shape))
    print("    The number of total different visual states: "+ str(len(data)))
    print("    " + str(args.seen) + "% visual states will be included in training process.")
    print("    " + str(100-args.seen) + "% states will be included in test process.")

    F_b, F_m = initialize_model(args.num, args.w_size)

    train(F_b, F_m, args.m, data, device, args.scene, args.num, args.seen, steps = 10000, bs = 512)
    
    if not os.path.exists(os.path.join(os.getcwd(), "output/")):
        os.makedirs(os.path.join(os.getcwd(), "output/"))

    th.save(F_b, 'output/CausalEncoder' +"_"+str(args.m)+"_"+args.scene+"_N"+str(args.num)+"_Size"+str(args.w_size)+ "_Seen"+str(args.seen))
    th.save(F_b, 'output/MomentumEncoder' +"_"+str(args.m)+"_"+args.scene+"_N"+str(args.num)+"_Size"+str(args.w_size)+ "_Seen"+str(args.seen))

    del F_b, F_m
    th.cuda.empty_cache()

if __name__ == '__main__':
    main()
