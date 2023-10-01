import numpy as np
import random
import math


def shear(data_numpy, r=0.5):
    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)
    return data_numpy


def temperal_crop(data_numpy, temperal_padding_ratio=6):
    C, T, V, M = data_numpy.shape
    padding_len = T // temperal_padding_ratio 
    frame_start = np.random.randint(0, padding_len * 2 + 1) 
    data_numpy = np.concatenate((data_numpy[:, :padding_len][:, ::-1],
                                 data_numpy,
                                 data_numpy[:, -padding_len:][:, ::-1]),
                                axis=1)
    data_numpy = data_numpy[:, frame_start:frame_start + T]
    return data_numpy


def camera_view(data_numpy, camera_view = 1):
    s = random.uniform(0.5*camera_view, 1.5*camera_view)
    r_list = [random.uniform(-camera_view*math.pi/6, camera_view*math.pi/6), 
              random.uniform(-camera_view*math.pi/6, camera_view*math.pi/6), 
              random.uniform(-camera_view*math.pi/6, camera_view*math.pi/6)]

    Rx = np.array([[1, 0, 0],
                  [0, math.cos(r_list[0]), -math.sin(r_list[0])],
                  [0, math.sin(r_list[0]), math.cos(r_list[0])]])
    Ry = np.array([[math.cos(r_list[1]), 0, math.sin(r_list[1])],
                  [0, 1, 0],
                  [-math.sin(r_list[1]), 0, math.cos(r_list[1])]])
    Rz = np.array([[math.cos(r_list[2]), -math.sin(r_list[2]), 0],
                  [math.sin(r_list[2]), math.cos(r_list[2]),0],
                  [0, 0, 1]])

    data_numpy = data_numpy * s
    data_numpy = np.einsum('bc,ctvm->btvm', Rx, data_numpy)
    #print(data_numpy)
    data_numpy = np.einsum('bc,ctvm->btvm', Ry, data_numpy)
    #print(data_numpy)
    data_numpy = np.einsum('bc,ctvm->btvm', Rz, data_numpy)
    #print(data_numpy)
    return data_numpy

def motion_scale(data_numpy, motion_scale=1):
    s = random.uniform(-2, 3)
    mp = (data_numpy[:, 0:48, :, :] + data_numpy[:, 2:50, :, :]) / 2
    d = data_numpy[:, 1:49, :, :] - mp
    newp = mp + s * d
    data_numpy[:, 1:49, :, :] = newp
    return data_numpy