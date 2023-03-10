# MoSN

Code and Environments for "A Self-Supervised Learning Solution with Momentum-Based Siamese Network for Robotic Visual Task".  

## Environments

We test the performance of MoSN in real-world environments, they are derived from a typical conference hall that can accommodate about 200 listeners.

And we evaluate MoSN in both static and dynamic environments with both fixed and dynamic viewing angles.


#### The Original Data

For each environment, we collect snapshots under all possible switch states. 

If you want to get these original data or have any other problems, feel free to send an email to fulva.hyh@gmail.com.

#### The Processed Data

To simplify our experiments, all the data is processed and stored as NPY files. 

For example:  RE-FA_all_images_N5_Size32.npy:

- SE-FA (a static real-world environment with a fixed viewing angle) is the name of the environment
- N is the number of switches and N5  means that the number of switches is 5 
- Size is the size of the input image and Size32 means that the size of our input image is (32, 32, 3) 

We could read our NPY files in the following way:

```python
data_dict = np.load(os.path.join(dirpath, args.scene+"_all_images_N"+str(args.num)+"_Size"+str(args.w_size)+".npy"), allow_pickle=True).item()
```

data_dict is a dict. The key is the switch state and the value is the image data.

For a dynamic environment such as DE-FA (a dynamic environment with a fixed viewing angle), data is stored as a array because one cause state can have several visual states. 

If you want to test MoSN in your environment，you could transfer your data to this format.



## Model Training & Evaluation

Train model with 30% snapshots under environment SE-FA with 7 switches 

```python
python main.py --m 0.999 --w-size 32 --num 7 --seen 30 --scene SE-FA
```

It will output training and test results every 1000 epochs. 

