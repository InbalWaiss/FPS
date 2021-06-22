import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import skimage.graph as sg


def build_map(n,m):
    my_map = np.zeros((m,n))
    my_map[5,3:8] = 1
    my_map[5:10, 10] = 1
    my_map[11,2:7] = 1
    my_map[:5,12] = 1
    blue = [2, 2]
    red = [5, 8]
    return my_map, blue, red

def calc_possible_locs(my_map, opponent_loc, depth = 10, neighborhood=4):
    queue = []
    queue.append(opponent_loc)
    res = my_map.copy()
    opponent_loc = np.asarray(opponent_loc)
    res[opponent_loc[0],opponent_loc[1]] = 2
    neighbors = np.asarray([[0,-1],[0,1],[-1,0],[1,0]])
    if neighborhood == 8:
        neighbors = np.asarray([[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[1,-1],[-1,1],[1,1]])
    while queue:
        curr = queue.pop(0)
        for nei_dir in neighbors:
            nei = curr + nei_dir
            if nei[0] < 0 or nei[0] >= my_map.shape[0] or nei[1] < 0 or nei[1] >= my_map.shape[1]:
                continue
            if res[nei[0], nei[1]]:
                continue
            if res[curr[0], curr[1]] < depth:
                res[nei[0], nei[1]] = res[curr[0], curr[1]]+1
                queue.append(nei)
                # plt.imshow(res)
                a=1
    return res


def calc_covers_map(my_map, red, depth=10):
    possible_locs = calc_possible_locs(my_map, red, depth=depth, neighborhood=8)
    without_obs = calc_possible_locs(np.zeros_like(my_map), red, depth=depth, neighborhood=8)
    diff = possible_locs - without_obs
    diff[my_map==1] = 0
    diff[diff < 0] = 0
    mx = diff.max()
    covers_map = diff == mx
    return covers_map, possible_locs, without_obs


def make_3d_map(map_2d, blue, red, depth):
    obstacles_map = map_2d==1
    map_3d = np.repeat(obstacles_map[:, :, np.newaxis], depth, axis=2)
    for i in range(depth):
        map_3d[:,:,i] = map_2d <= i+2
    return map_3d
    # show_3d_as_video(map_3d, case="map")
    # map_3d_0_path = mapper.embed_2d_path_in_3d_map(map_3d, path_2d_rob_0, l)

def embed_path_in_3d_map(map_3d, robot_0_path, l, enum=4):
    map_3d_0_path = map_3d.copy()
    for i in range(len(robot_0_path)):
        loc = robot_0_path[i]
        map_3d_0_path[loc[0]-l:loc[0]+l, loc[1]-l:loc[1]+l, loc[2]] = enum
    return map_3d_0_path


def show_3d_as_video(map_3d, s0=None, t0=None, s1=None, t1=None, case=""):
    for i in range(map_3d.shape[2]):
        im = map_3d[:, :, i]
        plt.clf()
        plt.imshow(im*255)
        if s0:
            plt.plot(s0[1], s0[0], 'og')
            plt.plot(t0[1], t0[0], 'xg')
            plt.plot(s1[1], s1[0], 'oy')
            plt.plot(t1[1], t1[0], 'xy')
        plt.title(case + " " + str(i))
        plt.pause(0.01)


def find_path_to_cover(map_3d, blue, red, closest_cover, time_to_cover):
    weights = map_3d.copy()
    weights[weights==0] = 1
    weights[weights == 1] = 1000
    path_3d, _ = sg.route_through_array(weights, [blue[0], blue[1], 0], [closest_cover[0], closest_cover[1], int(time_to_cover-1)], geometric=True)
    map_with_path = embed_path_in_3d_map(map_3d.astype(float), path_3d, 1, enum=4)
    show_3d_as_video(map_with_path,case='map_with_path')
    return path_3d


def select_cover(covers_map, without_obs, blue, red, depth=10):
    cands = np.where(covers_map == covers_map.max())
    blue = np.asarray(blue)
    min_dist = 100000
    closest = []
    for i in range(cands[0].shape[0]):
        cand = np.asarray([cands[0][i], cands[1][i]])
        dist = np.linalg.norm(blue - cand)
        if dist < min_dist:
            min_dist = dist
            closest = cand
    return closest
    # covers_map_ = covers_map.copy().astype(float)
    # if blue[1] < red[1]:
    #     covers_map_[:, red[1]+1:] = 5
    # else:
    #     covers_map_[:, :red[1]] = 5
    # if blue[0] < red[0]:
    #     covers_map_[red[0]+1:, :] = 5
    # else:
    #     covers_map_[:red[0], :] = 5
    # if covers_map_.any():
    #     ## find closest cover to target:
    #     without_obs[~covers_map] = depth+1
    #     cover = np.where(without_obs == np.min(without_obs))
    #     cover = [cover[0][0], cover[1][0]]
    #     return cover
    # else:
    #     covers_map_ = covers_map.copy().astype(float)

def main():
    my_map, blue, red = build_map(15, 15)
    depth = 10
    covers_map, possible_locs, without_obs= calc_covers_map(my_map, red, depth=depth)
    closest_cover = select_cover(covers_map, without_obs, blue, red)
    map_3d = make_3d_map(possible_locs,  blue, red, depth=depth)
    time_to_cover = possible_locs[closest_cover[0], closest_cover[1]]
    find_path_to_cover(map_3d, blue, red, closest_cover, time_to_cover)
    a=1


if __name__ == '__main__':
    main()
