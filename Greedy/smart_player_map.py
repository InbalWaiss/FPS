import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import skimage.graph as sg
from Common.constants import *


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


def calc_covers_map_old(my_map, red, depth=10):
    possible_locs = calc_possible_locs(my_map, red, depth=depth, neighborhood=8)
    without_obs = calc_possible_locs(np.zeros_like(my_map), red, depth=depth, neighborhood=8)
    diff = possible_locs - without_obs
    diff[my_map == 100] = 0
    diff[diff < 0] = 0
    mx = diff.max()
    covers_map = diff == mx
    return covers_map, possible_locs, without_obs


def calc_covers_map(my_map, enemy):
    enemy = np.asarray(enemy)
    fire_map = np.zeros_like(my_map)
    for j in [0, my_map.shape[1]]:
        for i in range(my_map.shape[0]):
            #update_clear_range(my_map, np.asarray([i, j]), enemy, depth, fire_map)
            mark_beyond_clear_range(my_map, np.asarray([i, j]), enemy, fire_map)
    for i in [0, my_map.shape[0]]:
        for j in range(my_map.shape[1]):
            # update_clear_range(my_map, np.asarray([i, j]), enemy, depth, fire_map)
            mark_beyond_clear_range(my_map, np.asarray([i, j]), enemy, fire_map)
    return fire_map


def update_clear_range(obs_map, to_loc, from_loc, depth, fire_map):
    dist = np.linalg.norm(to_loc - from_loc)
    step = 1./dist
    from_loc = np.asarray(from_loc)
    for α in np.arange(0, 1, step):
        loc = np.abs((1-α)*from_loc + α*to_loc).astype(int)
        dist = np.linalg.norm(from_loc - loc)
        if dist > depth or obs_map[loc[0], loc[1]]:
            return
        else:
            fire_map[loc[0], loc[1]] = True

def mark_beyond_clear_range(obs_map, to_loc, from_loc, fire_map):
    dist = np.linalg.norm(to_loc - from_loc)
    step = 1./dist
    met_wall = False
    for α in np.arange(0, 1, step):
        loc = np.abs((1-α)*from_loc + α*to_loc).astype(int)
        if met_wall and not obs_map[loc[0], loc[1]]:
            met_wall = False
            fire_map[loc[0], loc[1]] = 1
        if obs_map[loc[0], loc[1]]:
            met_wall = True


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
    path_3d, _ = sg.route_through_array(weights, [blue[0], blue[1], 0], [closest_cover[0], closest_cover[1], int(weights.shape[2]-1)], geometric=True)
    # map_with_path = embed_path_in_3d_map(map_3d.astype(float), path_3d, 1, enum=4)
    # show_3d_as_video(map_with_path,case='map_with_path')
    return path_3d


def select_cover(covers_map, without_obs, blue, red, depth=10):
    cands = np.where(covers_map > 0)
    blue = np.asarray(blue)
    min_dist = 100000
    closest = []
    for i in range(cands[0].shape[0]):
        cand = np.asarray([cands[0][i], cands[1][i]])
        dist = np.linalg.norm(blue - cand, 1)
        if dist < min_dist and dist < covers_map[cand[0], cand[1]]:
            min_dist = dist
            closest = cand
    return closest


def select_cover_old(covers_map, without_obs, blue, red, depth=10):
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


def is_clear_range(reach_time_loc, cover, obs_map):
    dist = np.linalg.norm(reach_time_loc - cover)
    step = 1./dist
    for α in np.arange(0, 1, step):
        loc = np.abs((1-α)*reach_time_loc + α*cover).astype(int)
        if obs_map[loc[0], loc[1]]:
            return False
    return True


def update_killing_range(covers_map, possible_locs, killing_range):
    # for all covers, calc when they will be in kiling range of a possible enemy loc
    covers = np.where(covers_map)
    n_covers = len(covers[0])
    obs_map = possible_locs == 100
    for cover_i in range(n_covers):
        cover = np.asarray([covers[0][cover_i], covers[1][cover_i]])
        final_reach_time = int(possible_locs[cover[1], cover[0]])
        reach_time = 0
        for reach_time in range(2, final_reach_time):
            reach_time_locs = np.where(possible_locs == reach_time)
            n_reach_time_locs = len(reach_time_locs[0])
            found = False
            for reach_time_loc_i in range(n_reach_time_locs):
                reach_time_loc = np.asarray([reach_time_locs[0][reach_time_loc_i], reach_time_locs[1][reach_time_loc_i]])
                if is_clear_range(reach_time_loc, cover, obs_map):
                    found = True
                    break
            if found:
                break
        covers_map[cover[0], cover[1]] = reach_time
    return covers_map


def find_move_in_path(player_path):
    i = 0
    while i < len(player_path)-1 and player_path[i][:2] == player_path[i+1][:2] :
        i += 1
    if i < len(player_path)-1:
        return player_path[i+1]
    else:
        return player_path[i]

def plan_next_action(state):
    future_length = 7
    my_pos = state.my_pos.get_tuple()
    enemy_pos = state.enemy_pos.get_tuple()
    im = state.env
    fire = im[:, :, 0] > im[:, :, 1] + 50
    # maybe fire is near and we can win:
    if fire[my_pos[0] - 1:my_pos[0] + 1, my_pos[1] - 1:my_pos[1] + 1].any():
        for i in range(-1, 2):
            broke = False
            for j in range(-1, 2):
                loc = [my_pos[0] + i, my_pos[1] + j]
                if loc[0] < 0 or loc[0] >= fire.shape[0] or loc[1] < 0 or loc[1] >= fire.shape[1]:
                    continue
                if fire[loc[0], loc[1]]:
                    action = get_action_9_actions(i, j, my_pos, fire.shape)
                    broke = True
                    break
            if broke:
                break
    else:  # search for a cover:
        im[im[:, :, 0] != im[:, :, 1]] = 0
        im[im[:, :, 2] != im[:, :, 1]] = 0
        killing_range = FIRE_RANGE
        my_path = plan_path(im[:, :, 0], my_pos, enemy_pos, future_length)
        if my_path:
            next_step = find_move_in_path(my_path)
            direc = np.asarray(next_step[:2]) - my_pos
            print("Go to cover: ", str(next_step[:2]), str(direc))
        else:  # no cover, just run far from enemy:
            direc = (np.asarray(my_pos) - np.asarray(enemy_pos))
            direc = direc / np.linalg.norm(direc)
            direc = np.round(direc)
            print("Run form enemy:", direc)

        action = get_action_9_actions(direc[0], direc[1], my_pos, fire.shape)
    return action

def get_action_9_actions( delta_x, delta_y, loc, shape):
    """9 possible moves!"""
    if loc[0] + delta_x < 0 or loc[0] + delta_x > shape[0]:
        delta_x = 0
    if loc[1] + delta_y < 0 or loc[1] + delta_y > shape[1]:
        delta_y = 0

    if delta_x == 1 and delta_y == -1:
        a = AgentAction.TopRight
    elif delta_x == 1 and delta_y == 0:
        a = AgentAction.Right
    elif delta_x == 1 and delta_y == 1:
        a = AgentAction.BottomRight
    elif delta_x == 0 and delta_y == -1:
        a = AgentAction.Bottom
    elif delta_x == 0 and delta_y == 1:
        a = AgentAction.Top
    elif delta_x == -1 and delta_y == -1:
        a = AgentAction.BottomLeft
    elif delta_x == -1 and delta_y == 0:
        a = AgentAction.Left
    elif delta_x == -1 and delta_y == 1:
        a = AgentAction.TopLeft
    else: ## delta_x == 0 and delta_y == 0:
        a = AgentAction.Stay

    return a

def plan_path(my_map, blue, red, depth):
    covers_map = calc_covers_map(my_map, red)
    possible_locs = calc_possible_locs(my_map, red, depth=FIRE_RANGE, neighborhood=8)
    possible_locs[possible_locs == 0] = FIRE_RANGE+1
    without_obs = calc_possible_locs(np.zeros_like(my_map), red, depth=FIRE_RANGE, neighborhood=8)
    covers_map = update_killing_range(covers_map.astype(int), possible_locs, FIRE_RANGE)

    closest_cover = select_cover(covers_map, without_obs, blue, red)

    if len(closest_cover):
        if (np.asarray(blue) == closest_cover).all():
            return [[closest_cover[0]],[closest_cover[0]]]
        map_3d = make_3d_map(possible_locs, blue, red, depth=depth)
        time_to_cover = possible_locs[closest_cover[1], closest_cover[0]]
        path = find_path_to_cover(map_3d, blue, red, closest_cover, time_to_cover)
        return path
    else:
        return []


def main():
    my_map, blue, red = build_map(15, 15)
    depth = 10
    killing_range = 8
    path = plan_path(my_map, blue, red, depth, killing_range)
    a=1


if __name__ == '__main__':
    main()
