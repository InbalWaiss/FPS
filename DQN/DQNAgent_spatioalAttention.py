
from Common.constants import *
from Arena.AbsDecisionMaker import AbsDecisionMaker
import os

from keras import backend as K

import argparse
import matplotlib.pyplot as plt

from DQN.deeprl_prj.policy import *
from DQN.deeprl_prj.objectives import *
from DQN.deeprl_prj.preprocessors import *
from DQN.deeprl_prj.utils import *
from DQN.deeprl_prj.core import  *
from DQN.helper import updateTargetGraph, updateTarget

REPLAY_MEMORY_SIZE = 50000 # how many last samples to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000 # minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64 # how many samples to use for training
UPDATE_TARGET_EVERY = 15 # number of terminal states
OBSERVATION_SPACE_VALUES = (SIZE_X, SIZE_Y, 3)
MODEL_NAME = 'SA_16(3X3)X32(3X1)X9_2'


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

class Qnetwork():
    def __init__(self, args, h_size, num_frames, num_actions, rnn_cell_1, myScope, rnn_cell_2=None):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.imageIn = tf.placeholder(shape=[None, 15, 15, num_frames], dtype=tf.float32)
        self.image_permute = tf.transpose(self.imageIn, perm=[0, 3, 1, 2])
        self.image_reshape = tf.reshape(self.image_permute, [-1, 15, 15, 1])
        self.image_reshape_recoverd = tf.squeeze(
            tf.gather(tf.reshape(self.image_reshape, [-1, num_frames, 15, 15, 1]), [0]), [0])
        self.summary_merged = tf.summary.merge(
            [tf.summary.image('image_reshape_recoverd', self.image_reshape_recoverd, max_outputs=num_frames)])
        # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,15,15,1])
        # self.conv1 = tf.contrib.layers.convolution2d( \
        #     inputs=self.image_reshape, num_outputs=32, \
        #     kernel_size=[8, 8], stride=[4, 4], padding='VALID', \
        #     activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv1')
        # self.conv2 = tf.contrib.layers.convolution2d( \
        #     inputs=self.conv1, num_outputs=64, \
        #     kernel_size=[4, 4], stride=[2, 2], padding='VALID', \
        #     activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv2')
        # self.conv3 = tf.contrib.layers.convolution2d( \
        #     inputs=self.conv2, num_outputs=64, \
        #     kernel_size=[3, 3], stride=[1, 1], padding='VALID', \
        #     activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv3')
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.image_reshape, num_outputs=32, \
            kernel_size=[3, 3], stride=[3, 3], padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv1')
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1, num_outputs=64, \
            kernel_size=[3, 3], stride=[1, 1], padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv2')
        self.conv4 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv2), h_size,
                                                       activation_fn=tf.nn.relu)

        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levels.
        # inbal # self.batch_size = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        self.convFlat = tf.reshape(self.conv4, [self.batch_size, num_frames, h_size])
        self.state_in_1 = rnn_cell_1.zero_state(self.batch_size, tf.float32)

        if args.bidir:
            self.state_in_2 = rnn_cell_2.zero_state(self.batch_size, tf.float32)
            self.rnn_outputs_tuple, self.rnn_state = tf.nn.bidirectional_dynamic_rnn( \
                cell_fw=rnn_cell_1, cell_bw=rnn_cell_2, inputs=self.convFlat, dtype=tf.float32, \
                initial_state_fw=self.state_in_1, initial_state_bw=self.state_in_2, scope=myScope + '_rnn')
            # print "====== len(self.rnn_outputs_tuple), self.rnn_outputs_tuple[0] ", len(self.rnn_outputs_tuple), self.rnn_outputs_tuple[0].get_shape().as_list(), self.rnn_outputs_tuple[1].get_shape().as_list() # [None, 10, 512]
            # As we have Bi-LSTM, we have two output, which are not connected. So merge them
            self.rnn_outputs = tf.concat([self.rnn_outputs_tuple[0], self.rnn_outputs_tuple[1]], axis=2)
            # self.rnn_outputs = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.rnn_outputs_double), h_size, activation_fn=None)
            self.rnn_output_dim = h_size * 2
        else:
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn( \
                inputs=self.convFlat, cell=rnn_cell_1, dtype=tf.float32, \
                initial_state=self.state_in_1, scope=myScope + '_rnn')
            # print "====== self.rnn_outputs ", self.rnn_outputs.get_shape().as_list() # [None, 10, 512]
            self.rnn_output_dim = h_size

        # attention machanism
        if not (args.a_t):
            self.rnn_last_output = tf.slice(self.rnn_outputs, [0, num_frames - 1, 0], [-1, 1, -1])
            self.rnn = tf.squeeze(self.rnn_last_output, [1])
        else:
            if args.global_a_t:
                self.rnn_outputs_before = tf.slice(self.rnn_outputs, [0, 0, 0], [-1, num_frames - 1, -1])
                self.attention_v = tf.reshape(tf.slice(self.rnn_outputs, [0, num_frames - 1, 0], [-1, 1, -1]),
                                              [-1, self.rnn_output_dim, 1])
                self.attention_va = tf.tanh(tf.matmul(self.rnn_outputs_before, self.attention_v))
                self.attention_a = tf.nn.softmax(self.attention_va, dim=1)
                self.rnn = tf.reduce_sum(tf.multiply(self.rnn_outputs_before, self.attention_a), axis=1)
                self.rnn = tf.concat(
                    [self.rnn, tf.squeeze(tf.slice(self.rnn_outputs, [0, num_frames - 1, 0], [-1, 1, -1]), [1])],
                    axis=1)
            else:
                with tf.variable_scope(myScope + '_attention'):
                    self.attention_v = tf.get_variable(name='atten_v', shape=[self.rnn_output_dim, 1],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                self.attention_va = tf.tanh(tf.map_fn(lambda x: tf.matmul(x, self.attention_v), self.rnn_outputs))
                self.attention_a = tf.nn.softmax(self.attention_va, dim=1)
                self.rnn = tf.reduce_sum(tf.multiply(self.rnn_outputs, self.attention_a), axis=1)
        # print "========== self.rnn ", self.rnn.get_shape().as_list() #[None, 1024]

        if args.net_mode == "duel":
            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.ad_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu,
                                                               scope=myScope + '_fc_advantage_hidden')
            self.Advantage = tf.contrib.layers.fully_connected(self.ad_hidden, num_actions, activation_fn=None,
                                                               scope=myScope + '_fc_advantage')
            self.value_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu,
                                                                  scope=myScope + '_fc_value_hidden')
            self.Value = tf.contrib.layers.fully_connected(self.value_hidden, 1, activation_fn=None,
                                                           scope=myScope + '_fc_value')
            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        else:
            self.Qout = tf.contrib.layers.fully_connected(self.rnn, num_actions, activation_fn=None)

        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class decision_maker_DQN_spatioalAttention:
    def __init__(self, path_model_to_load=None):
        self._previous_stats = {}
        self._action = {}
        # self._epsilon = epsilon
        self.model = None
        self.target_model = None

        self.is_training = IS_TRAINING
        self.numberOfSteps_allTournament = 0
        self.burn_in = True

        self.episode_number = 0
        self.episode_loss = 0
        self.episode_target_value = 0

        self._Initialize_networks(path_model_to_load)


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
        parser.add_argument('--exploration_steps', default=1000000, type=int, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
        parser.add_argument('--num_samples', default=100000000, type=int, help='Number of training samples from the environment in training')
        parser.add_argument('--num_frames', default=1, type=int, help='Number of frames to feed to Q-Network')
        parser.add_argument('--frame_width', default=15, type=int, help='Resized frame width')
        parser.add_argument('--frame_height', default=15, type=int, help='Resized frame height')
        parser.add_argument('--replay_memory_size', default=1000000, type=int, help='Number of replay memory the agent uses for training')
        parser.add_argument('--target_update_freq', default=50000, type=int, help='The frequency with which the target network is updated')
        parser.add_argument('--train_freq', default=4, type=int, help='The frequency of actions wrt Q-network update')
        parser.add_argument('--save_freq', default=50000, type=int, help='The frequency with which the network is saved')
        parser.add_argument('--eval_freq', default=50000, type=int, help='The frequency with which the policy is evlauted')
        parser.add_argument('--num_burn_in', default=10000, type=int,
                            help='Number of steps to populate the replay memory before training starts')
        parser.add_argument('--load_network', default=False, action='store_true', help='Load trained mode')
        parser.add_argument('--load_network_path', default='', help='the path to the trained mode file')
        parser.add_argument('--net_mode', default='dqn', help='choose the mode of net, can be linear, dqn, duel')
        parser.add_argument('--max_episode_length', default = 10000, type=int, help = 'max length of each episode')
        parser.add_argument('--num_episodes_at_test', default = 20, type=int, help='Number of episodes the agent plays at test')
        parser.add_argument('--ddqn', default=False, dest='ddqn', action='store_true', help='enable ddqn')
        parser.add_argument('--train', default=True, dest='train', action='store_true', help='Train mode')
        parser.add_argument('--test', dest='train', action='store_false', help='Test mode')
        parser.add_argument('--no_experience', default=False, action='store_true', help='do not use experience replay')
        parser.add_argument('--no_target', default=False, action='store_true', help='do not use target fixing')
        parser.add_argument('--no_monitor', default=False, action='store_true', help='do not record video')
        parser.add_argument('--task_name', default='SpatialAt_DQN', help='task name')
        parser.add_argument('--recurrent', default=True, dest='recurrent', action='store_true', help='enable recurrent DQN')
        parser.add_argument('--a_t', default=True, dest='a_t', action='store_true', help='enable temporal/spatial attention')
        parser.add_argument('--global_a_t', default=False, dest='global_a_t', action='store_true', help='enable global temporal attention')
        parser.add_argument('--selector', default=True, dest='selector', action='store_true', help='enable selector for spatial attention')
        parser.add_argument('--bidir', default=False, dest='bidir', action='store_true', help='enable two layer bidirectional lstm')

        args = parser.parse_args()

        return args

    def _set_previous_state(self, state):
        self._previous_stats = state

    def _set_epsilon(self, input_epsilon):
        # self._epsilon = input_epsilon
        pass

    def reset_networks(self):
        self._Initialize_networks()

    def _Initialize_networks(self, path_model_to_load = None):
        args = self.get_args()

        self.num_actions = NUMBER_OF_ACTIONS
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        self.history_processor = HistoryPreprocessor(args.num_frames - 1)
        self.atari_processor = AtariPreprocessor()
        self.memory = ReplayMemory(args)
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
        self.output_path_videos = args.output + '/videos/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.net_mode
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        self.args = args

        self.h_size = 512
        self.tau = 0.001
        # self.q_network = create_model(input_shape, num_actions, self.net_mode, args, "QNet")
        # self.target_network = create_model(input_shape, num_actions, self.net_mode, args, "TargetNet")
        tf.reset_default_graph()
        # We define the cells for the primary and target q-networks
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        cellT = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        if args.bidir:
            cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
            cellT_2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
            self.q_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames,
                                      num_actions=self.num_actions, rnn_cell_1=cell, rnn_cell_2=cell_2,
                                      myScope="QNet")
            self.target_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames,
                                           num_actions=self.num_actions, rnn_cell_1=cellT, rnn_cell_2=cellT_2,
                                           myScope="TargetNet")
        else:
            self.q_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames,
                                      num_actions=self.num_actions, rnn_cell_1=cell, myScope="QNet")
            self.target_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames,
                                           num_actions=self.num_actions, rnn_cell_1=cellT, myScope="TargetNet")

        # initialize target network
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        trainables = tf.trainable_variables()
        print(trainables, len(trainables))
        self.targetOps = updateTargetGraph(trainables, self.tau)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        updateTarget(self.targetOps, self.sess)
        self.writer = tf.summary.FileWriter(self.output_path)

        if path_model_to_load!=None:
            # path_model_to_load = 'statistics/15_02_21_29_DQNAgent_spatioalAttention_Q_table/qnet660000.cptk'
            self.saver.restore(self.sess, path_model_to_load)
            print("+++++++++ Network restored from: %s", path_model_to_load)

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
            # action_mask = np.zeros((batch_size, self.num_actions))
            # action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            # next_qa_value = self.target_network.predict_on_batch(next_states)
            next_qa_value = self.sess.run(self.target_network.Qout, \
                                          feed_dict={self.target_network.imageIn: next_states,
                                                     self.target_network.batch_size: batch_size})

        if self.enable_ddqn:
            # qa_value = self.q_network.predict_on_batch(next_states)
            qa_value = self.sess.run(self.q_network.Qout, \
                                     feed_dict={self.q_network.imageIn: next_states,
                                                self.q_network.batch_size: batch_size})
            max_actions = np.argmax(qa_value, axis=1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis=1)
        # print rewards.shape, mask.shape, next_qa_value.shape, batch_size
        target = rewards + self.gamma * mask * next_qa_value

        if self.args.a_t and np.random.random() < 1e-3:
            loss, _, rnn, attention_v, attention_a = self.sess.run(
                [self.q_network.loss, self.q_network.updateModel, self.q_network.rnn, self.q_network.attention_v,
                 self.q_network.attention_a], \
                feed_dict={self.q_network.imageIn: states, self.q_network.batch_size: batch_size, \
                           self.q_network.actions: actions, self.q_network.targetQ: target})
            # print(attention_a[0])
        else:
            loss, _, rnn = self.sess.run([self.q_network.loss, self.q_network.updateModel, self.q_network.rnn], \
                                         feed_dict={self.q_network.imageIn: states,
                                                    self.q_network.batch_size: batch_size, \
                                                    self.q_network.actions: actions, self.q_network.targetQ: target})

        return loss, np.mean(target)


    def _get_action(self, current_state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        dqn_state = current_state.img
        policy_type = "UniformRandomPolicy" if self.burn_in else "LinearDecayGreedyEpsilonPolicy"
        state_for_network = self.atari_processor.process_state_for_network(dqn_state)
        action_state = self.history_processor.process_state_for_network(state_for_network)
        q_values = self.calc_q_values(action_state)

        if self.is_training:
            if policy_type == 'UniformRandomPolicy':
                action =  UniformRandomPolicy(self.num_actions).select_action()
            else:
                # linear decay greedy epsilon policy
                action =  self.policy.select_action(q_values, self.is_training)
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            action =  GreedyPolicy().select_action(q_values)


        # self._epsilon = max([self._epsilon * EPSILONE_DECAY, 0.05])  # change epsilon
        self._action = action
        return action

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        # return self.q_network.predict_on_batch(state)
        # print state.shape
        # Qout = self.sess.run(self.q_network.rnn_outputs,\
        #             feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        Qout = self.sess.run(self.q_network.Qout, \
                             feed_dict={self.q_network.imageIn: state, self.q_network.batch_size: 1})
        # print Qout.shape
        return Qout


    # adds step's data to memory replay array
    # (state, action, reward, new_state, is_terminal)
    def update_replay_memory(self, transition):
        self.memory.append(transition[0], transition[1], transition[2], transition[4])



    def train(self, new_state, reward, is_terminal):

        self.numberOfSteps_allTournament+=1
        # if not self.burn_in: #the replay buffer has at least num_burn_in samples
            # self.frames+=1
            # self.episode_raw_reward += reward

        if is_terminal:
            # adding last frame only to save last state
            last_frame = self.atari_processor.process_state_for_memory(new_state)
            self.memory.append(last_frame, self._action, reward, is_terminal) #TODO in original code it was (last_frame, action, 0, is_terminal)- why 0?
            if not self.burn_in:
                pass

            self.burn_in = (self.numberOfSteps_allTournament < self.num_burn_in)
            self.atari_processor.reset()
            self.history_processor.reset()

        if not self.burn_in:
            if self.numberOfSteps_allTournament % self.train_freq == 0:
                action_state = self.history_processor.process_state_for_network(
                    self.atari_processor.process_state_for_network(new_state))
                processed_reward = self.atari_processor.process_reward(reward)
                processed_next_state = self.atari_processor.process_state_for_network(new_state)
                action_next_state = np.dstack((action_state, processed_next_state))
                action_next_state = action_next_state[:, :, 1:]
                current_sample = Sample(action_state, self._action, processed_reward, action_next_state, is_terminal)
                loss, target_value = self.update_policy(current_sample)
                self.episode_loss += loss
                self.episode_target_value += target_value

        # update freq is based on train_freq
        if self.numberOfSteps_allTournament % (self.train_freq * self.target_update_freq) == 0:
            # target updates can have the option to be hard or soft
            # related functions are defined in deeprl_prj.utils
            # here we use hard target update as default
            updateTarget(self.targetOps, self.sess)
            print("----- Synced.")

        self._previous_stats = new_state
        # self.burn_in = (self.numberOfSteps_allTournament < self.num_burn_in) #inbal: update burn_in flag always or just in terminal state?

        # if is_terminal:
        #     self.episode_number += 1



    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        #return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape) / 255)[0]
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape))[0]

    def save_model(self, idx_episode, path):
        safe_path = path + "/qnet" + str(idx_episode) + ".cptk"
        self.saver.save(self.sess, safe_path)
        print("+++++++++ Network at", idx_episode, "saved to:", safe_path)

    def restore_model(self, restore_path):
        self.saver.restore(self.sess, restore_path)
        print("+++++++++ Network restored from: %s", restore_path)

    def print_model(self, state, episode_number, path_to_dir):
        path = os.path.join(path_to_dir, str(episode_number))
        if not os.path.exists(path):
            os.makedirs(path)

        dqn_state = state.img
        state_for_network = self.atari_processor.process_state_for_network(dqn_state)
        action_state = self.history_processor.process_state_for_network(state_for_network)

        # q_values = self.calc_q_values(action_state) #shold be action_state

        # save image
        plt.figure()
        plt.imshow(dqn_state)
        p = os.path.join(path, 'start_state_img.png')
        plt.imsave(p, dqn_state, format='png')
        plt.close()

        inp_conv_1 = self.target_network.conv1
        outputs = [layer.output for layer in self.target_network.layers]

        inp = self.target_network.input  # input placeholder
        outputs = [layer.output for layer in self.target_network.layers]  # all layer outputs
        functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function

        t = (action_state)[np.newaxis, ...]
        layer_outs = functor([t, 1.])

        plt.figure()
        for ind_layer in range(0,4):
            p = os.path.join(path, 'layer_'+str(ind_layer))
            if not os.path.exists(p):
                os.makedirs(p)
            layer = layer_outs[ind_layer]
            for filter_index in range(layer.shape[-1]):
                filter = layer[0,:,:,filter_index]
                # shape_x = filter.shape[1]
                # shape_y = filter.shape[2]
                # img = out.reshape(shape_x,shape_y)
                # plt.imshow(filter)
                file_name = os.path.join(p, 'filter_'+ str(filter_index)+'.png')
                plt.imsave(file_name, filter, format='png')

        plt.close()

# Agent class
class DQNAgent_spatioalAttention(AbsDecisionMaker):
    def __init__(self, UPDATE_CONTEXT = True, path_model_to_load=None):
        self._previous_state = None
        self._action = None
        self.episode_number = 0
        self._decision_maker = decision_maker_DQN_spatioalAttention(path_model_to_load)
        self.min_reward = -np.Inf
        self._type = AgentType.DQNAgent_spatioalAttention
        self.path_model_to_load = path_model_to_load
        self.UPDATE_CONTEXT =UPDATE_CONTEXT

    def type(self) -> AgentType:
        return self._type

    def set_initial_state(self, initial_state_blue, episode_number):
        self.episode_number = episode_number
        self._previous_state = initial_state_blue

    def get_action(self, current_state):
        action = self._decision_maker._get_action(current_state)
        self._action = AgentAction(action)
        return self._action

    def update_context(self, new_state, reward, is_terminal):
        if self.UPDATE_CONTEXT:
            previous_state_for_network = self._decision_maker.atari_processor.process_state_for_memory(self._previous_state)
            new_state_for_network = self._decision_maker.atari_processor.process_state_for_memory(new_state)
            transition = (previous_state_for_network, self._action, reward, new_state_for_network, is_terminal)
            self._decision_maker.update_replay_memory(transition)
            self._decision_maker.train(new_state, reward, is_terminal)
            self._previous_state = new_state

    def get_epsolon(self):
        if IS_TRAINING:
            return self._decision_maker.policy.epsilon
        else:
            return 0

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
        self._decision_maker.save_model(self.episode_number, path_to_model)
        # self._decision_maker.q_network.save(
        #     f'{path_to_model+os.sep+MODEL_NAME}_{color_str}_{NUM_OF_EPISODES}_{max_reward: >7.2f}max_{avg_reward: >7.2f}avg_{min_reward: >7.2f}min__{int(time.time())}.model')

        return self.min_reward

