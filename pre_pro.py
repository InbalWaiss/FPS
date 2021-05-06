import matplotlib.pyplot as plt
import networkx as nx
from Arena.Position import Position
from Arena.CState import State
from Arena.AbsDecisionMaker import AbsDecisionMaker
from Common.constants import *
import numpy as np
import os

PRINT_FLAG = False


class Greedy_player(AbsDecisionMaker):

    def __init__(self, UPDATE_CONTEXT=True , path_model_to_load=None):

        self._action = -1
        self._type = AgentType.Greedy
        self.episode_number = 0
        self._epsilon = 0
        self.path_model_to_load = None

        self.G = self.create_graph()

        self.add_to_all_pairs_distances = False
        self.add_to_all_pairs_shortest_path = False
        self.add_to_closest_target_dict = False
        self.all_pairs_distances = {}
        self.all_pairs_shortest_path = {}
        self.closest_target_dict = {}


    def create_graph(self):
        G = nx.grid_2d_graph(SIZE_X, SIZE_Y)
        pos = dict((n, n) for n in G.nodes())  # Dictionary of all positions
        labels = dict(((i, j), (i, j)) for i, j in G.nodes())

        if NUMBER_OF_ACTIONS >= 8:
            Diagonals_Weight = 1
            # add diagonals edges
            G.add_edges_from([
                                 ((x, y), (x + 1, y + 1))
                                 for x in range(SIZE_Y - 1)
                                 for y in range(SIZE_Y - 1)
                             ] + [
                                 ((x + 1, y), (x, y + 1))
                                 for x in range(SIZE_Y - 1)
                                 for y in range(SIZE_Y - 1)
                             ], weight=Diagonals_Weight)

        # remove obstacle nodes and edges
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    G.remove_node((x, y))

        # self.all_pairs_distances = dict(nx.floyd_warshall(G))
        # self.all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(G))

        # nx.write_gpickle(G, 'Greedy_'+DSM_name+'.pkl')
        # loaded_G = nx.read_gpickle('Greedy_'+DSM_name+'.pkl')


        if PRINT_FLAG:
            path = self.all_pairs_shortest_path[(3,10)][(10,3)]
            path_edges = set(zip(path, path[1:]))
            nx.draw_networkx(G, pos=pos, labels=labels, font_size=5, with_labels=True, node_size=50)

            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='g')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color='b')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[1]], node_color='black')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color='r')
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='black', width=3)
            plt.axis('equal')
            plt.show()

        return G



    def calc_all_pairs_data(self, DSM):
        SIZE_X = 100
        SIZE_Y = 100
        G = nx.grid_2d_graph(SIZE_X, SIZE_Y)

        if NUMBER_OF_ACTIONS >= 8:
            Diagonals_Weight = 1
            # add diagonals edges
            G.add_edges_from([
                                 ((x, y), (x + 1, y + 1))
                                 for x in range(SIZE_Y - 1)
                                 for y in range(SIZE_Y - 1)
                             ] + [
                                 ((x + 1, y), (x, y + 1))
                                 for x in range(SIZE_Y - 1)
                                 for y in range(SIZE_Y - 1)
                             ], weight=Diagonals_Weight)

        # remove obstacle nodes and edges
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    G.remove_node((x, y))

        # nx.write_gpickle(G, 'G_' + DSM_name + '.pkl')

        import bz2
        print("starting all_pairs_distances")
        all_pairs_distances = dict(nx.all_pairs_shortest_path_length(G))

        with bz2.open('all_pairs_distances_' + DSM_name + '___' + '.pkl', "wb") as f:
            f.write(all_pairs_distances)
        # sfile = bz2.BZ2File('all_pairs_distances_' + DSM_name + '___' + '.pkl', 'wb', 'w')
        # pickle.dump(all_pairs_distances, sfile)
        print("finished all_pairs_distances")

        with bz2.open('all_pairs_distances_' + DSM_name + '___' + '.pkl', "rb") as f:
            # Decompress data from file
            content = f.read()
        print("dist from (5,5) to (5,6) is: ", content[(5,5)][(5,5)])

        # hugeData = {'key': {'x': 1, 'y': 2}}
        # with contextlib.closing(bz2.BZ2File('data.json.bz2', 'wb')) as f:
        #     json.dump(hugeData, f)
        #
        # with open('all_pairs_shortest_path_' + DSM_name + '___' + '.pkl', 'wb') as f:
        #     pickle.dump(all_pairs_shortest_path, f, protocol=2)
        # print("finished all_pairs_shortest_path")
        #
        #
        # #loaded_G = nx.read_gpickle('Greedy_'+DSM_name+'.pkl')


    def remove_data_obs(self, DSM):
        all_pairs_distances_path = 'all_pairs_distances_100X100_Berlin___.pkl'
        if os.path.exists(all_pairs_distances_path):
            with open(all_pairs_distances_path, 'rb') as f:
                all_pairs_distances = pickle.load(f)
                print("Greedy: all_pairs_distances_100X100_Berlin___ loaded")

        filtered_data = {}
        for p1 in all_pairs_distances.keys():
            for p2 in all_pairs_distances[p1].keys():
                if DSM[p1]==0 and DSM[p2]==0:
                    if not p1 in filtered_data.keys():
                        filtered_data[p1]={}
                    filtered_data[p1][p2] = all_pairs_distances[p1][p2]

        with open('all_pairs_distances_' + DSM_name + '___filtered' + '.pkl', 'wb') as f:
            pickle.dump(filtered_data, f,  protocol=2)



if __name__ == '__main__':
    #PRINT_FLAG = True
    from PIL import Image
    import cv2
    srcImage = Image.open("Common/maps/Berlin_1_256.png")

    img1 = np.array(srcImage.convert('L').resize((100, 100)))
    img2 = cv2.bitwise_not(img1)
    obsticals = cv2.inRange(img2, 250, 255)
    c, _ = cv2.findContours(obsticals, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thicken_obs_and_edges = cv2.drawContours(obsticals, c, -1, (255, 255, 255), 2)
    thicken_obs_and_edges[thicken_obs_and_edges > 0] = 1
    DSM = thicken_obs_and_edges

    if False:
        plt.matshow(thicken_obs_and_edges)
        plt.show(DSM)

    GP = Greedy_player()
    #GP.remove_data_obs(DSM)
    GP.calc_all_pairs_data(DSM)

    # blue_pos = Position(3, 10)
    # red_pos = Position(10, 3)
    # ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)
    #
    # a = GP.get_action(ret_val)
    # print("The action to take is: ", a)
