from Common.constants import *
from DQN.DQN_constants import *
import os
import time
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras import backend as K


REPLAY_MEMORY_SIZE = 50000 # how many last samples to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100 # minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64 # how many samples to use for training
UPDATE_TARGET_EVERY = 15 # number of terminal states
if BB_STATE:
    OBSERVATION_SPACE_VALUES = (SIZE_X_BB, SIZE_Y_BB, 3)
else:
    OBSERVATION_SPACE_VALUES = (SIZE_X, SIZE_Y, 3)
MODEL_NAME = 'red_blue_32(4X4)X64X9'


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overrided. saves logs with our step number. otherwise every .fit() will start writing from 0th step
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided. we train for one batch only, no need to save anythings at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so wont close writer
    def on_train_end(self, _):
        pass

    # custom method for saving metrics
    # creats writer, write custom metrics and close writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        # with self.writer.as_default():
        #     for name, value in logs.items():
        #         with self.writer.as_default():
        #             tf.summary.scalar(name, value, step=index)
        #         self.step += 1
        #         self.writer.flush()

        # for name, value in logs.items():
        #     with self.writer:
        #         tf.summary.scalar(name, value)
        #         self.step += 1
        #         self.writer.flush()

        pass

class decision_maker_DQN():
    def __init__(self, path_model_to_load=None):
        self._previous_stats = {}
        self._action = {}
        self._epsilon = epsilon
        self.model = None
        self.target_model = None
        self.target_update_counter = 0

        self._Initialize_networks(path_model_to_load)

    def _set_previous_state(self, state):
        self._previous_stats = state

    def _set_epsilon(self, input_epsilon):
        self._epsilon = input_epsilon

    def get_layer_outputs(self, state):

        outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        comp_graph = [K.function([self.model.input] + [K.learning_phase()], [output]) for output in
                      outputs]  # evaluation functions

        # Testing
        layer_outputs_list = [op([state, 1.]) for op in comp_graph]
        layer_outputs = []

        for layer_output in layer_outputs_list:
            print(layer_output[0][0].shape, end='\n-------------------\n')
            layer_outputs.append(layer_output[0][0])

        return layer_outputs

    def print_model(self, state, episode_number, path_to_dir):
        img = np.array(state.img).reshape(-1, *np.array(state.img).shape) / 255
        path = os.path.join(path_to_dir, str(episode_number))
        if not os.path.exists(path):
            os.makedirs(path)

        layer_outputs = self.get_layer_outputs(img)

        layer_number = 0

        x_max = layer_outputs[layer_number].shape[0]
        y_max = layer_outputs[layer_number].shape[1]
        n = layer_outputs[layer_number].shape[2]

        L = []
        for i in range(n):
            L.append(np.zeros((x_max, y_max)))

        for i in range(n):
            for x in range(x_max):
                for y in range(y_max):
                    L[i][x][y] = layer_outputs[layer_number][x][y][i]


        plt.figure()
        index = 0
        for img in L:
            # plt.imshow(img, interpolation='nearest')
            p = os.path.join(path, 'img_'+ str(index)+'.png')
            plt.imsave(p, img, format='png')
            index+=1
        # plt.figure()
        # plt.imshow(L[00])
        plt.close()


        # save image
        plt.figure()
        org_img = img[0, :, :, :]
        # plt.imshow(org_img)
        p = os.path.join(path, 'state_img.png')
        plt.imsave(p, org_img, format='png')
        plt.close()

        # cv2.imwrite('D:\\RL\\layer0.jpg', L[00])
        # plt.imsave('D:\\RL\\layer22.png', L[22], format='png')

    def reset_networks(self):
        self._Initialize_networks()

    def _Initialize_networks(self, path_model_to_load = None):
        # load model
        if path_model_to_load !=None:
            p = path.join(RELATIVE_PATH_HUMAN_VS_MACHINE_DATA, path_model_to_load)
            self.model = load_model(p)
            self.target_model = load_model(p)
            self.target_model.set_weights(self.model.get_weights())

        else: #create new model
            self.model = self.create_model() # main model
            self.target_model = self.create_model() # target model

        # array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # custom tesnsorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))


    def create_model_conv(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))


        model.add(Flatten()) # this converts out 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(NUMBER_OF_ACTIONS, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    def create_model(self):
        model = Sequential()
        model.add(Dense(128, activation="elu", input_shape=OBSERVATION_SPACE_VALUES))


        model.add(Flatten())  # this converts out 3D feature maps to 1D feature vectors
        model.add(Dense(128, activation="elu"))
        model.add(Dense(128, activation="elu"))
        model.add(Dense(128, activation="elu"))

        model.add(Dense(NUMBER_OF_ACTIONS, activation='elu'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def loadModel(self, model, target_model):
        self.model = model
        self.target_model = target_model



    def update_replay_memory(self, transition):
        # adds step's data to memory replay array
        # (state, action, reward, new_state, is_terminal)
        self.replay_memory.append(transition)


    def train(self, terminal_state, step):
        # Trains main network every step during episode
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current state from minibatch, then query NN model for Q values
        current_state = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_state)

        # Get future states from minibatch, the query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_state = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.model.predict(new_current_state)

        X=[]
        Y=[]

        # Now we need to enumerate out batches
        for index, (current_state, action, reward, new_current_state, is_terminal) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q learning, but we use just part of the equation here
            if not is_terminal:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to out training data
            X.append(current_state)
            Y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        # start = time.time()
        self.model.fit(np.array(X) / 255, np.array(Y), batch_size = MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # endt = time.time()
        # print('Fitting time == ', endt-start)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape) / 255)[0]

    def _get_action(self, current_state, is_training=False, EVALUATE=False):
        dqn_state = current_state.img
        if np.random.random() > self._epsilon or not is_training or EVALUATE:
            # Get action from network
            action = np.argmax(self.get_qs(dqn_state))
        else:
            # Get random action
            action = np.random.randint(0, NUMBER_OF_ACTIONS)

        self._action = action
        return action

    def update_epsilon(self):
        self._epsilon = max([self._epsilon * EPSILONE_DECAY, min_epsilon])  # change epsilon

# Agent class
class DQNAgent():
    def __init__(self, UPDATE_CONTEXT=True, path_model_to_load=None):
        self._previous_state = None
        self._action = None
        self.episode_number = 0
        self._decision_maker = decision_maker_DQN(path_model_to_load)
        self.min_reward = -np.Inf
        self._type = AgentType.DQN_basic
        self.path_model_to_load = path_model_to_load
        self.is_training = IS_TRAINING
        self.UPDATE_CONTEXT = UPDATE_CONTEXT

    def type(self) -> AgentType:
        return self._type

    def set_initial_state(self, initial_state_blue, episode_number):
        self.episode_number = episode_number
        self._previous_state = initial_state_blue
        self._decision_maker.tensorboard.step = episode_number
        self._decision_maker.update_epsilon()
        pass

    def get_action(self, current_state, EVALUATE=False):
        action = self._decision_maker._get_action(current_state=current_state, is_training=self.is_training, EVALUATE=EVALUATE)
        self._action = AgentAction(action)
        return self._action

    def get_epsolon(self):
        return self._decision_maker._epsilon

    def update_context(self, new_state, reward, is_terminal, EVALUATE=False):

        if not self.is_training or not self.UPDATE_CONTEXT:# or not EVALUATE:
            return

        transition = (self._previous_state.img, self._action, reward, new_state.img, is_terminal)
        self._decision_maker.update_replay_memory(transition)
        self._decision_maker.train(is_terminal, self.episode_number)

        self._previous_state = new_state


    def save_model(self, ep_rewards, path_to_model, player_color):

        avg_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        min_reward = min(ep_rewards[-SHOW_EVERY:])
        max_reward = max(ep_rewards[-SHOW_EVERY:])
        #TODO: uncomment this! # self._decision_maker.tensorboard.update_state(reward_avg = avg_reward, reward_min = min_reward, reward_max = max_reward, epsilon = epsilon)

        episode = len(ep_rewards)
        # save model, but only when min reward is greater or equal a set value
        if max_reward >=self.min_reward or episode == NUM_OF_EPISODES-1:
            self.min_reward = min_reward
            if player_color == Color.Red:
                color_str = "red"
            elif player_color == Color.Blue:
                color_str = "blue"
            self._decision_maker.model.save(
                f'{path_to_model+os.sep+MODEL_NAME}_{color_str}_{NUM_OF_EPISODES}_{max_reward: >7.2f}max_{avg_reward: >7.2f}avg_{min_reward: >7.2f}min__{int(time.time())}.model')

        return self.min_reward
