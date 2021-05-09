
from Common.constants import *
from DQN.DQN_constants import *
import os
import time
from keras.models import load_model

from keras.optimizers import (Adam)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          merge, Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute)
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

import argparse
import matplotlib.pyplot as plt

from DQN.deeprl_prj.policy import *
from DQN.deeprl_prj.objectives import *
from DQN.deeprl_prj.preprocessors import *
from DQN.deeprl_prj.utils import *
from DQN.deeprl_prj.core import  *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))


def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)

class decision_maker_DQN_keras:
    def __init__(self, path_model_to_load=None):
        self._previous_stats = {}

        self._epsilon = START_EPSILON
        self.model = None
        self.target_model = None

        self.q_network = None
        self.target_network = None

        self.is_training = IS_TRAINING
        self.numberOfSteps = 0
        self.burn_in = True

        self.episode_number = 0
        self.episode_loss = 0
        self.episode_target_value = 0

        self._Initialize_networks(path_model_to_load)

        self.MODEL_NAME = self.set_model_name()

        # save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH,
        #                              format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
        #                                     f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}"))
        #
        # self.writer = tf.summary.FileWriter(save_folder_path)


    def get_args(self):
        parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
        parser.add_argument('--env', default='shoot me if you can', help='small world')
        parser.add_argument('-o', '--output', default='./log/', help='Directory to save data to')
        parser.add_argument('--seed', default=0, type=int, help='Random seed')
        parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')

        parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial exploration probability in epsilon-greedy')
        parser.add_argument('--final_epsilon', default=0.05, type=float, help='Final exploration probability in epsilon-greedy')

        parser.add_argument('--num_samples', default=100000000, type=int, help='Number of training samples from the environment in training')
        parser.add_argument('--num_frames', default=NUM_FRAMES, type=int, help='Number of frames to feed to Q-Network')

        if BB_STATE:
            parser.add_argument('--frame_width', default=SIZE_X_BB, type=int, help='Resized frame width')
            parser.add_argument('--frame_height', default=SIZE_Y_BB, type=int, help='Resized frame height')
            parser.add_argument('--exploration_steps', default=7000000, type=int,
                                help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
        else:
            parser.add_argument('--frame_width', default=SIZE_X, type=int, help='Resized frame width')
            parser.add_argument('--frame_height', default=SIZE_Y, type=int, help='Resized frame height')
            parser.add_argument('--exploration_steps', default=7000000, type=int,
                                help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')


        parser.add_argument('--replay_memory_size', default=500000, type=int, help='Number of replay memory the agent uses for training')
        parser.add_argument('--target_update_freq', default=5000, type=int, help='The frequency with which the target network is updated')
        parser.add_argument('--train_freq', default=1, type=int, help='The frequency of actions wrt Q-network update')
        parser.add_argument('--save_freq', default=50000, type=int, help='The frequency with which the network is saved')
        parser.add_argument('--eval_freq', default=50000, type=int, help='The frequency with which the policy is evlauted')
        parser.add_argument('--num_burn_in', default=10000, type=int,
                            help='Number of steps to populate the replay memory before training starts')
        parser.add_argument('--load_network', default=False, action='store_true', help='Load trained mode')
        parser.add_argument('--load_network_path', default='', help='the path to the trained mode file')

        if FULLY_CONNECTED:
            parser.add_argument('--net_mode', default='linear', help='choose the mode of net, can be linear, dqn, duel')
        else:
            parser.add_argument('--net_mode', default='dqn', help='choose the mode of net, can be linear, dqn, duel')


        parser.add_argument('--max_episode_length', default = 10000, type=int, help = 'max length of each episode')
        parser.add_argument('--num_episodes_at_test', default = 20, type=int, help='Number of episodes the agent plays at test')
        parser.add_argument('--ddqn', default=True, dest='ddqn', action='store_true', help='enable ddqn')
        parser.add_argument('--train', default=True, dest='train', action='store_true', help='Train mode')
        parser.add_argument('--test', dest='train', action='store_false', help='Test mode')
        parser.add_argument('--no_experience', default=False, action='store_true', help='do not use experience replay')
        parser.add_argument('--no_target', default=False, action='store_true', help='do not use target fixing')
        parser.add_argument('--no_monitor', default=False, action='store_true', help='do not record video')
        parser.add_argument('--task_name', default='', help='task name')
        parser.add_argument('--recurrent', default=False, dest='recurrent', action='store_true', help='enable recurrent DQN')
        parser.add_argument('--a_t', default=False, dest='a_t', action='store_true', help='enable temporal/spatial attention')
        parser.add_argument('--global_a_t', default=False, dest='global_a_t', action='store_true', help='enable global temporal attention')
        parser.add_argument('--selector', default=False, dest='selector', action='store_true', help='enable selector for spatial attention')
        parser.add_argument('--bidir', default=False, dest='bidir', action='store_true', help='enable two layer bidirectional lstm')

        args = parser.parse_args()

        return args



    def _set_previous_state(self, state):
        self._previous_stats = state

    def _set_epsilon(self, input_epsilon):
        self._epsilon = input_epsilon

    def reset_networks(self):
        self._Initialize_networks()

    def _Initialize_networks(self, path_model_to_load = None):
        # load model
        args = self.get_args()
        self.num_actions = NUMBER_OF_ACTIONS
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        self.history_processor = HistoryPreprocessor(args.num_frames - 1)
        self.atari_processor = AtariPreprocessor()
        self.memory = ReplayMemory(args)
        self.exploration_steps = args.exploration_steps
        self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon,
                                                     args.exploration_steps)
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        self.num_burn_in = args.num_burn_in
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_frames = args.num_frames
        self.output_path = args.output
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.net_mode
        print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
        self.eval_freq = args.eval_freq
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        if path_model_to_load !=None:
            p = path.join(RELATIVE_PATH_HUMAN_VS_MACHINE_DATA, path_model_to_load)
            self.q_network = load_model(p)
            self.target_network = load_model(p)
            self.target_network.set_weights(self.q_network.get_weights())
            self.final_model = self.target_network
            self.compile()

        else: #create new model
            self.q_network = self.create_model(input_shape, self.num_actions, self.net_mode, args, "QNet")
            self.target_network = self.create_model(input_shape, self.num_actions, self.net_mode, args, "TargetNet")

            # initialize target network
            self.target_network.set_weights(self.q_network.get_weights())
            self.final_model = None
            self.compile()

        self.writer = tf.summary.FileWriter(self.output_path)

    def set_model_name(self):
        s = ''
        for i in range(1, len(self.q_network.layers) - 1):
            layer = self.q_network.layers[i]
            s = s + str(layer.name)
            if isinstance(layer, Convolution2D):
                layer_shape = layer.kernel.shape
                s = s + '(' + str(layer_shape[0]) + '_' + str(layer_shape[1]) + '_' + str(layer_shape[2]) + '_' + str(
                    layer_shape[3]) + ')'
            s = s + '_'
        return s

    def loadModel(self, model, target_model):
        # load existing models
        self.model = model
        self.target_model = target_model

    def create_model(self, input_shape, num_actions, mode, args, model_name='q_network'):
        """Create the Q-network model.

        Use Keras to construct a keras.models.Model instance.

        Parameters
        ----------
        window: int
          Each input to the network is a sequence of frames. This value
          defines how many frames are in the sequence.
        input_shape: tuple(int, int, int), rows, cols, channels
          The expected input image size.
        num_actions: int
          Number of possible actions. Defined by the gym environment.
        model_name: str
          Useful when debugging. Makes the model show up nicer in tensorboard.

        Returns
        -------
        keras.models.Model
          The Q-model.
        """
        assert (mode in ("linear", "duel", "dqn"))
        with tf.variable_scope(model_name):
            input_data = Input(shape=input_shape, name="input")
            if mode == "linear":

                flatten_hidden = Flatten(name="flatten")(input_data)
                FC_1 = Dense(512, activation='elu', name='FC1-elu')(flatten_hidden)
                FC_2 = Dense(512, activation='elu', name='FC2-elu')(FC_1)
                FC_3 = Dense(512, activation='elu', name='FC3-elu')(FC_2)
                FC_4 = Dense(512, activation='elu', name='FC4-elu')(FC_3)
                output = Dense(num_actions, activation='elu', name="output")(FC_4)

            else:
                if not (args.recurrent):
                    if DSM_name=="15X15":
                        h1 = Convolution2D(64, (4, 4), strides=2, activation="relu", name="conv2")(input_data)
                        h2 = Convolution2D(64, (3, 3), strides=1, activation="relu", name="conv3")(h1)
                        context = Flatten(name="flatten")(h2)

                    else:
                        # # version 1:
                        #h1 = Convolution2D(32, (8, 8), strides=4, activation="relu", name="conv1")(input_data)
                        h1 = Convolution2D(256, (6, 6), strides=3, activation="relu", name="conv1")(input_data)
                        h2 = Convolution2D(128, (4, 4), strides=2, activation="relu", name="conv2")(h1)
                        h3 = Convolution2D(128, (3, 3), strides=1, activation="relu", name="conv3")(h2)
                        context = Flatten(name="flatten")(h3)


                if mode == "dqn":
                    h4 = Dense(512, activation='elu', name="fc")(context)
                    output = Dense(num_actions, name="output")(h4)

        model = Model(inputs=input_data, outputs=output)
        print(model.summary())
        return model


    def compile(self, optimizer = None, loss_func = None):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is the place to create the target network, setup
        loss function and any placeholders.
        """
        if loss_func is None:
            #loss_func = mean_huber_loss
            loss_func = 'mse'
        if optimizer is None:
            optimizer = Adam(lr = self.learning_rate)
            # optimizer = RMSprop(lr=0.00025)
        with tf.variable_scope("Loss"):
            state = Input(shape = (self.frame_height, self.frame_width, self.num_frames) , name = "states")
            action_mask = Input(shape = (self.num_actions,), name = "actions")
            qa_value = self.q_network(state)
            qa_value = merge([qa_value, action_mask], mode = 'mul', name = "multiply")
            qa_value = Lambda(lambda x: tf.reduce_sum(x, axis=1, keep_dims = True), name = "sum")(qa_value)

        #loss_func = losses.mean_squared_error
        self.final_model = Model(inputs = [state, action_mask], outputs = qa_value)
        self.final_model.compile(loss=loss_func, optimizer=optimizer)



    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        return self.q_network.predict_on_batch(state)

    def train(self, state, action, reward, new_state, is_terminal):

        self.numberOfSteps += 1

        if is_terminal:
            # adding last frame only to save last state
            # last_frame_state = self.atari_processor.process_state_for_memory(state)
            # last_frame_new_state = self.atari_processor.process_state_for_memory(new_state)
            #self.memory.append(last_frame_state, action, reward, last_frame_new_state, is_terminal)
            # self.atari_processor.reset()
            # self.history_processor.reset()
            if not self.burn_in:
                self.episode_reward = .0
                self.episode_raw_reward = .0
                self.episode_loss = .0
                self.episode_target_value = .0

        if not self.burn_in: # enough samples in replay buffer
            if self.numberOfSteps % self.train_freq == 0:
                # action_state = self.history_processor.process_state_for_network(self.atari_processor.process_state_for_network(state))
                # processed_reward = self.atari_processor.process_reward(reward)
                # processed_next_state = self.atari_processor.process_state_for_network(new_state)
                # action_next_state = np.dstack((action_state, processed_next_state))
                # action_next_state = action_next_state[:, :, 1:]
                # current_sample = Sample(action_state, int(action), processed_reward, action_next_state, is_terminal)


                # #inbal: before multi frams
                # last_frame_state = self.atari_processor.process_state_for_memory(state)
                # last_frame_new_state = self.atari_processor.process_state_for_memory(new_state)
                # current_sample = Sample(last_frame_state, int(action), reward, last_frame_new_state, is_terminal)
                #

                current_sample = []
                loss, target_value = self.update_policy(current_sample)
                self.episode_loss += loss
                self.episode_target_value += target_value

            # update freq is based on train_freq
            if self.numberOfSteps % (self.train_freq * self.target_update_freq) == 0:
                # target updates can have the option to be hard or soft
                # related functions are defined in deeprl_prj.utils
                # here we use hard target update as default
                self.target_network.set_weights(self.q_network.get_weights())



        self._previous_stats = new_state
        self.burn_in = (self.numberOfSteps < self.num_burn_in)

        if is_terminal:
            self.episode_number += 1

    def print_frames(self, last_frame_state, last_frame_new_state=None):
        import matplotlib.pyplot as plt

        plt.matshow(last_frame_state)
        plt.show()

        if last_frame_new_state is not None:
            plt.matshow(last_frame_new_state)
            plt.show()

    def update_policy(self, current_sample):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        # current_sample = current_sample.img
        batch_size = self.batch_size

        if self.no_experience:
            states = np.stack([current_sample.state])
            next_states = np.stack([current_sample.next_state])
            rewards = np.asarray([current_sample.reward])
            mask = np.asarray([1 - int(current_sample.is_terminal)])

            action_mask = np.zeros((1, self.num_actions))
            action_mask[0, current_sample.action] = 1.0
        else:
            samples = self.memory.sample(batch_size)
            samples = self.atari_processor.process_batch(samples)

            states = np.stack([x.state for x in samples])
            actions = np.asarray([x.action for x in samples])
            action_mask = np.zeros((batch_size, self.num_actions))
            action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            next_qa_value = self.target_network.predict_on_batch(next_states)

        if self.enable_ddqn:
            qa_value = self.q_network.predict_on_batch(next_states)
            max_actions = np.argmax(qa_value, axis = 1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]

        else:
            next_qa_value = np.max(next_qa_value, axis = 1)
        target = rewards + self.gamma * mask * next_qa_value

        return self.final_model.train_on_batch([states, action_mask], target), np.mean(target)

    # adds step's data to memory replay array
    # (state, action, reward, new_state, is_terminal)
    def update_replay_memory(self, state, action, reward, new_state, is_terminal):
        self.memory.append(state, action, reward, new_state, is_terminal)
        # if is_terminal: # adding last frame only to save last frame
        #     self.memory.append(new_state, action, CONST_LAST_FRAME_REWARD, new_state, is_terminal)
        #     self.atari_processor.reset()
        #     self.history_processor.reset()

    def _get_action(self, current_state, evaluate=False, **kwargs):
        dqn_state = current_state.img
        DEBUG=False
        if DEBUG:
            plt.matshow(dqn_state)
            plt.matshow(action_state)
            plt.matshow(action_state[:, :, 0])
            plt.matshow(action_state[:, :, 1])
            plt.matshow(action_state[:, :, 2])

        """Select the action based on the current state.

        Returns
        --------
        selected action
        """
        self.numberOfSteps += 1
        policy_type = "UniformRandomPolicy" if self.burn_in else "LinearDecayGreedyEpsilonPolicy"
        state_for_network = self.atari_processor.process_state_for_network(dqn_state)
        action_state = self.history_processor.process_state_for_network(state_for_network)

        action = None
        q_values = self.calc_q_values(action_state)
        if self.is_training and not evaluate:
            if policy_type == 'UniformRandomPolicy':
                action= UniformRandomPolicy(NUMBER_OF_ACTIONS).select_action()
                self._epsilon = 1
            else:
                # linear decay greedy epsilon policy
                action = self.policy.select_action(q_values, self.is_training)
                self._epsilon = self.policy.epsilon
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            action = GreedyPolicy().select_action(q_values)
            if not evaluate:
                self._epsilon = 0

        return action

    def print_model(self, state, episode_number, path_to_dir):
        from keras import backend as K
        path = os.path.join(path_to_dir, str(episode_number))
        if not os.path.exists(path):
            os.makedirs(path)

        dqn_state = state.img
        state_for_network = self.atari_processor.process_state_for_network(dqn_state)
        action_state = self.history_processor.process_state_for_network(state_for_network)

        # save image
        plt.figure()
        plt.imshow(dqn_state)
        p = os.path.join(path, 'start_state_img.png')
        plt.imsave(p, dqn_state, format='png')
        plt.close()

        inp = self.target_network.input  # input placeholder
        outputs = [layer.output for layer in self.target_network.layers]  # all layer outputs
        functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function

        t = (action_state)[np.newaxis, ...]
        layer_outs = functor([t, 1.])

        num_of_conv_layers = 2

        p_fram_number = os.path.join(path, 'frams')
        if not os.path.exists(p_fram_number):
            os.makedirs(p_fram_number)
        first_layer = layer_outs[0]
        for fram_ind in range(self.num_frames):
            fram = first_layer[0,:,:,fram_ind]
            fram_file_name = os.path.join(p_fram_number, 'filter_' + str(fram_ind) + '.png')
            plt.imsave(fram_file_name, fram, format='png')

        for ind_layer in range(0, len(layer_outs)):
            p_ind_layer = os.path.join(path, 'layer_' + str(ind_layer))
            if not os.path.exists(p_ind_layer):
                os.makedirs(p_ind_layer)
            layer = layer_outs[ind_layer]
            if len(np.shape(layer))==4:
                for filter_index in range(layer.shape[-1]):
                    fram = layer[:, :, :, filter_index]
                    p_fram_number = os.path.join(p_ind_layer, 'fram_' + str(ind_layer))
                    if not os.path.exists(p_fram_number):
                        os.makedirs(p_fram_number)
                    filter = fram[0, :]
                    file_name = os.path.join(p_fram_number, 'filter_' + str(filter_index) + '.png')
                    plt.imsave(file_name, filter, format='png')

        plt.close()


    def plot_cinv_weights(self):

        W = self.target_network.get_layer(name="conv1").get_weights()[0]
        if len(W.shape) == 4:
            # W = np.squeeze(W1)
            W1 = W.reshape((W.shape[0], W.shape[1], W.shape[2] * W.shape[3]))
            fig, axs = plt.subplots(8, 8, figsize=(8, 8))
            fig.subplots_adjust(hspace=.5, wspace=.001)
            axs = axs.ravel()
            for i in range(64):
                axs[i].imshow(W1[:, :, i])
                axs[i].set_title(str(i))


# Agent class
class DQNAgent_keras:
    def __init__(self, UPDATE_CONTEXT=True, path_model_to_load=None):


        self.episode_number = 0
        self._decision_maker = decision_maker_DQN_keras(path_model_to_load)
        self.min_reward = -np.Inf
        self._type = AgentType.DQN_keras
        self.path_model_to_load = path_model_to_load
        self.UPDATE_CONTEXT = UPDATE_CONTEXT

    def type(self) -> AgentType:
        return self._type

    def get_epsolon(self):
        return self._decision_maker._epsilon

    def set_initial_state(self, initial_state_blue, episode_number):
        self.episode_number = episode_number

        pass

    def get_action(self, current_state, EVALUATE=False):
        action = self._decision_maker._get_action(current_state, EVALUATE)
        return AgentAction(action)

    def update_context(self, state, action, reward, new_state, is_terminal, EVALUATE=True):
        previous_state_for_network = self._decision_maker.atari_processor.process_state_for_memory(state)
        new_state_for_network = self._decision_maker.atari_processor.process_state_for_memory(new_state)
        transition = (previous_state_for_network, action, reward, new_state_for_network, is_terminal)
        self._decision_maker.update_replay_memory(previous_state_for_network, action, reward, new_state_for_network,
                                                  is_terminal)
        if self.UPDATE_CONTEXT and not EVALUATE:
            self._decision_maker.train(state, action, reward, new_state, is_terminal)





    def save_model(self, ep_rewards, path_to_model, player_color):

        avg_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        min_reward = min(ep_rewards[-SHOW_EVERY:])
        max_reward = max(ep_rewards[-SHOW_EVERY:])
        #TODO: uncomment this! # self._decision_maker.tensorboard.update_state(reward_avg = avg_reward, reward_min = min_reward, reward_max = max_reward, epsilon = epsilon)

        episode = len(ep_rewards)
        # save model, but only when min reward is greater or equal a set value
        # if max_reward >=self.min_reward or episode == NUM_OF_EPISODES-1:
        self.min_reward = min_reward
        if player_color == Color.Red:
            color_str = "red"
        elif player_color == Color.Blue:
            color_str = "blue"
        self._decision_maker.q_network.save(
            f'{path_to_model+os.sep+self._decision_maker.MODEL_NAME}_{color_str}_{episode}_{max_reward: >7.2f}max_{avg_reward: >7.2f}avg_{min_reward: >7.2f}min__{int(time.time())}.model')

        return self.min_reward
