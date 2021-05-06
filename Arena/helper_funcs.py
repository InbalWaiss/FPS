import unittest

from Common.constants import *
from Arena.geometry import LOS, bresenham


def check_if_LOS(x1, y1, x2, y2):
    """returns True is there are no obstacles between (x1,y1) and (x2,y2)
    otherwise return False"""
    if USE_BRESENHAM_LINE:
        list_of_points = bresenham(x1, y1, x2, y2)
    else:
        list_of_points = LOS(x1, y1, x2, y2)

    points_in_los = []
    for [x, y] in list_of_points:
        if DSM[x, y] == 1:
            # print("Hit in: (", {x}, " ,", {y}, ")")
            return False, points_in_los
        else:
            points_in_los.append([x, y])
    return True, points_in_los


# can the first player escape the second
def can_escape(first_player, second_player):

    org_cor_first_player_x, org_cor_first_player_y = first_player.get_coordinates()
    org_cor_second_player_x, org_cor_second_player_y = second_player.get_coordinates()
    winnig_point = (-1, -1)
    ret_val = False
    for action in range(0, NUMBER_OF_ACTIONS):

        first_player.set_coordinatess(org_cor_first_player_x, org_cor_first_player_y)
        first_player.action(action)
        first_player_after_action_x, first_player_after_action_y = first_player.get_coordinates()

        is_los = (org_cor_second_player_x, org_cor_second_player_y) in DICT_POS_FIRE_RANGE[(first_player.x, first_player.y)]

        if not is_los:
            ret_val = True
            winnig_point = (first_player_after_action_x, first_player_after_action_y)
            break

    first_player.set_coordinatess(org_cor_first_player_x, org_cor_first_player_y)
    return ret_val, winnig_point










class Test_is_dominating(unittest.TestCase):
    # DSM = np.array([
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
    #     [0., 1., 1., 0., 1., 0., 0., 1., 0., 0.],
    #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ])

    def test_controling1(self):
        blue_player = Blue_Entity()
        blue_player.x = 1
        blue_player.y = 3

        red_player = Red_Entity()
        red_player.x = 4
        red_player.y = 2

        self.assertFalse(is_dominating(blue_player, red_player))
        if not is_dominating(blue_player, red_player):
            print("Pass controlingTest1")
        else:
            print("WTF")

    def test_controling2(self):
        blue_player = Blue_Entity()
        blue_player.x = 2
        blue_player.y = 3

        red_player = Red_Entity()
        red_player.x = 4
        red_player.y = 2

        self.assertTrue(is_dominating(blue_player, red_player))
        if is_dominating(blue_player, red_player):
            print("Pass controlingTest2")