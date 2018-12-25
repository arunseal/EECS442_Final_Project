from torch.autograd import Function, Variable
import numpy as np
import torch
from layers import MAELoss
import cv2







def main():

    # perpendicular vectors, answer should be pi/2
    # a = [[[[1,1,1],
    #       [1,1,1],
    #       [1,1,1]],
    #      [[0,0,0],
    #       [0,0,0],
    #       [0,0,0]],
    #      [[0,0,0],
    #       [0,0,0],
    #       [0,0,0]]]]
    # a = np.array(a).astype(np.float32)
    # a = torch.from_numpy(a)


    # b = [[[[0,0,0],
    #       [0,0,0],
    #       [0,0,0]],
    #      [[1,1,1],
    #       [1,1,1],
    #       [1,1,1]],
    #      [[0,0,0],
    #       [0,0,0],
    #       [0,0,0]]]]
    # b = np.array(b).astype(np.float32)
    # b = torch.from_numpy(b)



    a = torch.from_numpy(np.random.rand(25,3,128,128)*255.0)
    b = torch.from_numpy(np.random.rand(25,3,128,128)*255.0)
    a,b = Variable(a, requires_grad=True),Variable(b, requires_grad=True)


    loss_fn = MAELoss()
 
    loss = loss_fn(a,b)
    print(loss)

    loss.backward()
    #print(back)










if __name__ == '__main__':
    main()