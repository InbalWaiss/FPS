
from Common.constants import *

class Evaluation():
    def __init__(self):
        self.evaluation__number_of_steps_batch = []
        self.evaluation__win_array_batch = []
        self.evaluation__rewards_for_blue_batch = []
        self.evaluation__epsilon_value_batch = []

        self.evaluation__number_of_steps = []
        self.evaluation__win_array_blue = []
        self.evaluation__win_array_tie = []
        self.evaluation__rewards_for_blue = []
        self.evaluation__epsilon_value = []


    def evaluate_info(self, EVALUATE_FLAG, episode_number, steps_current_game, blue_epsilon, win_status):

        if episode_number % EVALUATE_PLAYERS_EVERY==EVALUATE_BATCH_SIZE:
            self.evaluation__number_of_steps.append(np.mean(self.evaluation__number_of_steps_batch))
            self.evaluation__rewards_for_blue.append(np.mean(self.evaluation__rewards_for_blue_batch))
            self.evaluation__epsilon_value.append(np.mean(self.evaluation__epsilon_value_batch))

            win_array = np.array(self.evaluation__win_array_batch)
            win_array_blue = (win_array == WinEnum.Blue) * 100
            self.evaluation__win_array_blue.append(np.mean(win_array_blue))

            win_array_Tie = (win_array == WinEnum.Tie) * 100
            self.evaluation__win_array_tie.append(np.mean(win_array_Tie))

            print("\nEvaluation summury: num_episodes: ", episode_number, ", epsilon is: ", np.mean(self.evaluation__epsilon_value_batch))
            print("Avg number of steps: ",  np.mean(self.evaluation__number_of_steps_batch))
            print("Avg reward for Blue: ", np.mean(self.evaluation__rewards_for_blue_batch))
            print("Win % for Blue: ", np.mean(win_array_blue))

            self.evaluation__number_of_steps_batch = []
            self.evaluation__win_array_batch = []
            self.evaluation__rewards_for_blue_batch = []
            self.evaluation__epsilon_value_batch = []

        elif EVALUATE_FLAG:
            self.evaluation__number_of_steps_batch.append(steps_current_game)
            self.evaluation__win_array_batch.append(win_status)
            self.evaluation__rewards_for_blue_batch.append(self.episodes_rewards_blue[-1])
            self.evaluation__epsilon_value_batch.append(blue_epsilon)

        return