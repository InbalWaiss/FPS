from Arena.Position import Position
from Common.constants import *
from Arena.helper_funcs import check_if_LOS
import matplotlib.pyplot as plt


class State(object):

    def __init__(self, my_pos: Position, enemy_pos: Position, Red_won= False):
        self.Red_won = Red_won
        self.my_pos = my_pos
        self.enemy_pos = enemy_pos
        self.img = self.get_image()



    def get_image(self):
        env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8) # starts an rbg of small world

        if not self.enemy_pos is None:
            if LOS_PENALTY_FLAG:
                points_in_enemy_los = DICT_POS_LOS[(self.enemy_pos._x, self.enemy_pos._y)]
                for point in points_in_enemy_los:
                    env[point[0]][point[1]] = dict_of_colors_for_state[DARK_DARK_RED_N]

            if DANGER_ZONE_IN_STATE:
                points_in_enemy_los = DICT_POS_FIRE_RANGE[(self.enemy_pos._x, self.enemy_pos._y)]
                for point in points_in_enemy_los:
                    color = dict_of_colors_for_state[RED_N]
                    if NONEDETERMINISTIC_TERMINAL_STATE:
                        dist = np.linalg.norm(np.array(point) - np.array([self.enemy_pos._x, self.enemy_pos._y]))
                        dist_floor = np.floor(dist)
                        enemy_color = dict_of_colors_for_state[RED_N]
                        color = tuple(map(lambda i, j: int(i - j), enemy_color, (15 * dist_floor, 0, 0)))
                    env[point[0]][point[1]] = color  # dict_of_colors_for_state[DARK_RED_N]


            env[self.enemy_pos._x][self.enemy_pos._y] = dict_of_colors_for_state[RED_N]

        if not self.Red_won:
            env[self.my_pos._x][self.my_pos._y] = dict_of_colors_for_state[BLUE_N]

        if (not BB_STATE):
            for x in range(SIZE_X):
                for y in range(SIZE_Y):
                    if DSM[x][y] == 1.:
                        env[x][y] = dict_of_colors_for_state[GREY_N]
        else:
            start_x = np.max([0, self.my_pos._x - FIRE_RANGE - BB_MARGIN])
            end_x = np.min([self.my_pos._x + FIRE_RANGE + BB_MARGIN + 1, SIZE_X])
            start_y = np.max([0, self.my_pos._y - FIRE_RANGE - BB_MARGIN])
            end_y = np.min([self.my_pos._y + FIRE_RANGE + BB_MARGIN + 1, SIZE_Y])
            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if DSM[x][y] == 1.:
                        env[x][y] = dict_of_colors_for_state[GREY_N]

            BB_env = np.zeros((SIZE_X_BB, SIZE_Y_BB, 3), dtype=np.uint8) * 1  # obs=1


            if (self.my_pos._x - FIRE_RANGE - BB_MARGIN) >= 0:
                start_ind_x_BB = 0
            else:
                start_ind_x_BB = -(self.my_pos._x - FIRE_RANGE - BB_MARGIN)

            if (self.my_pos._x + FIRE_RANGE + BB_MARGIN) >= SIZE_X:
                end_ind_x_BB = (SIZE_X - 1) - (self.my_pos._x + FIRE_RANGE + BB_MARGIN)
            else:
                end_ind_x_BB = SIZE_X_BB

            if (self.my_pos._y - FIRE_RANGE - BB_MARGIN) >= 0:
                start_ind_y_BB = 0
            else:
                start_ind_y_BB = -(self.my_pos._y - FIRE_RANGE - BB_MARGIN)

            if (self.my_pos._y + FIRE_RANGE + BB_MARGIN) >= SIZE_Y:
                end_ind_y_BB = (SIZE_Y - 1) - (self.my_pos._y + FIRE_RANGE + BB_MARGIN)
            else:
                end_ind_y_BB = SIZE_Y_BB

            BB_env[start_ind_x_BB:end_ind_x_BB, start_ind_y_BB:end_ind_y_BB] = env[start_x:end_x, start_y:end_y]

            if False:
                plt.matshow(env)
                plt.show()
                plt.matshow(BB_env)
                plt.show()


            env = BB_env

        return env


def print_env(env):
    # print state for debug
    import matplotlib.pyplot as plt
    plt.matshow(env)
    plt.show()