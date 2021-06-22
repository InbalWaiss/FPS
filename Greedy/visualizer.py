import matplotlib.pyplot as plt
import numpy as np


def show_3d_as_video(map_3d, s0=None, t0=None, s1=None, t1=None):
    for i in range(map_3d.shape[2]):
        im = map_3d[:, :, i]
        plt.clf()
        plt.imshow(im)
        if s0:
            plt.plot(s0[1], s0[0], 'og')
        if t0:
            plt.plot(t0[1], t0[0], 'xg')
        if s1:
            plt.plot(s1[1], s1[0], 'oy')
        if t1:
            plt.plot(t1[1], t1[0], 'xy')
        plt.pause(0.01)


def show_3d_as_strip(map_3d, s0=None, t0=None, s1=None, t1=None):
    res = np.zeros((map_3d.shape[0], 0))
    for i in range(map_3d.shape[2]):
        im = map_3d[:, :, i]
        res = np.hstack((res, im))
    plt.imshow(res)
    plt.show()
    return res