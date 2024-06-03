import cProfile
import sys
from time import sleep
import gym
import numpy as np
from collections import deque
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from gym.wrappers import HumanRendering
from datetime import datetime
import pstats
import os
import gc
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'

warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

tf.debugging.disable_check_numerics()
tf.debugging.disable_traceback_filtering()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

envE = gym.make("Seaquest-v4", render_mode='rgb_array')
#envE.metadata['render_fps'] = 600
envE.reset()

# var = env.render()
#env3 = HumanRendering(envE)
#env3.reset()

#STATESIZE = (88, 80, 1)
STATESIZE = (88, 160, 1)
#STATESIZE = (210, 160, 1)
ACTION_SIZE = envE.action_space.n
MINIBATCH_SIZE = 32
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.9438182157859615

OUTPUTREWFILE = "rewData/"
OUTPUTOTHERFILE = "otherFiles/"
ACTIVATION = 'relu'
LAYERS_COUNT = 0

STARTING_GAME = 45

PREVACTION = 10

REPLCOUNT = 1
C_EPISODES = 40
C_STEPS = 1000

data = {
    'Reward': [],
    'Episode': [],
}

MIN_REWARD_TO_SAVE = 10
TRAINING_TIME = 0

#epsilon_decay_target = (epsilon_min / epsilon_start) ** (1 / episodes_target)

@tf.function
def preprocess_state(image):
    image = tf.cast(image, dtype=tf.float16) / 255
    image = tf.strided_slice(image, [1, 0, 0], [176, image.shape[1], image.shape[2]], [2, 1, 1])
    image = tf.image.rgb_to_grayscale(image)
    return image

class DQN:
    def __init__(self, state_size, action_size, createNN):
        #self.replay_buffer_dataset = None
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=100000)
        self.gamma = 0.98
        self.epsilon = 0.9
        self.update_rate = 1000
        if createNN:
            self.main_network1 = self.build_network()

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), strides=(4, 4), input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(Flatten())
        for i in range(LAYERS_COUNT):
            model.add(Dense(128, activation=ACTIVATION))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))
        return model

    def store_transition(self, state, action, reward, next_state, next_action, done):
        self.replay_buffer.append((state, action, reward, next_state, next_action, done))

    @tf.function
    def epsilon_greedy(self, state):
        if tf.random.uniform([], 0, 1) <= self.epsilon:
            return tf.random.uniform([], 0, self.action_size, dtype=tf.int32)
        with tf.device('/GPU:0'):
            stateT = tf.convert_to_tensor(state[None, :], dtype=tf.float16)
            Q_values = self.main_network1(stateT, training=False)
        return tf.argmax(Q_values[0], output_type=tf.int32)

    @tf.function
    def prepareData(self):
        # minibatch : 0 = states, 1 = actions, 2 = rewards, 3 = next_states, 4 = next_actions , 5 = dones
        transitions = list(self.replay_buffer)
        states, actions, rewards, next_states, next_actions, dones = map(tf.stack, zip(*transitions))
        indices = tf.random.shuffle(tf.range(len(transitions)))[:MINIBATCH_SIZE]

        minibatch_states = tf.gather(states, indices)
        minibatch_actions = tf.gather(actions, indices)
        minibatch_rewards = tf.gather(rewards, indices)
        minibatch_next_states = tf.gather(next_states, indices)
        minibatch_next_actions = tf.gather(next_actions, indices)
        minibatch_dones = tf.gather(dones, indices)

        predictsNextQ = self.main_network1(minibatch_next_states, training=False)
        nextActionQValues = tf.gather_nd(predictsNextQ, tf.stack((tf.range(MINIBATCH_SIZE), minibatch_next_actions), axis=1))

        notDones = 1.0 - tf.cast(minibatch_dones, tf.float32)
        targetQs = minibatch_rewards + self.gamma * nextActionQValues * notDones

        currentQs = self.main_network1(minibatch_states, training=False)
        updates = tf.stack([tf.range(MINIBATCH_SIZE), tf.cast(minibatch_actions, tf.int32)], axis=1)

        currentQsUpd = tf.tensor_scatter_nd_update(currentQs, updates, targetQs)

        return minibatch_states, currentQsUpd


    """ @tf.function pokus o spravenie datasetu --- neslo rychlejsie aj tak
    def prepareData(self):
        states = tf.stack([x[0] for x in self.replay_buffer])
        actions = tf.stack([x[1] for x in self.replay_buffer])
        rewards = tf.stack([x[2] for x in self.replay_buffer])
        next_states = tf.stack([x[3] for x in self.replay_buffer])
        dones = tf.stack([x[4] for x in self.replay_buffer])

        replay_buffer_dataset = tf.data.Dataset.from_tensor_slices({
            "states": states,
            "actions": tf.expand_dims(actions, -1),
            "rewards": tf.expand_dims(rewards, -1),
            "next_states": next_states,
            "dones": tf.expand_dims(dones, -1)
        })

        self.replay_buffer_dataset = replay_buffer_dataset.shuffle(100000).batch(MINIBATCH_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE)

        predicts_Q1 = self.target_network1(next_states, training=False)
        predicts_Q2 = self.target_network2(next_states, training=False)
        notDones = 1.0 - tf.cast(dones, tf.float32)
        targets_Q1 = rewards + self.gamma * tf.reduce_max(predicts_Q1, axis=1) * notDones
        targets_Q2 = rewards + self.gamma * tf.reduce_max(predicts_Q2, axis=1) * notDones

        Current_Qs1 = self.main_network1(states, training=False)
        Current_Qs2 = self.main_network2(states, training=False)
        indices = tf.stack([tf.range(tf.shape(states)[0]), tf.cast(actions, tf.int32)], axis=1)
        Current_Qs1 = tf.tensor_scatter_nd_update(Current_Qs1, indices, targets_Q1)
        Current_Qs2 = tf.tensor_scatter_nd_update(Current_Qs2, indices, targets_Q2)

        return states, Current_Qs1, Current_Qs2"""

    def train(self, states, cQs1):
        self.main_network1.train_on_batch(states, cQs1, reset_metrics=False)

def printPlot():
    df = pd.read_csv(OUTPUTREWFILE)
    plt.figure(figsize=(8, 5))
    plt.plot(df['Episode'], df['Reward'], marker='o', linestyle='-', color='b')
    plt.title('Reward vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

def saveData(TRAINING_TIME):
    df = pd.DataFrame(data)

    df.to_csv(OUTPUTREWFILE + "_Repl_" + str(REPLCOUNT) + "_ACTIVATION_" + ACTIVATION + "_LAYERS_COUNT_" + str(
        LAYERS_COUNT) + ".csv", index=False)
    print(TRAINING_TIME)
    TRAINING_TIME /= REPLCOUNT
    print(TRAINING_TIME)

    maxScore = np.max(data['Reward'])
    minScore = np.min(data['Reward'])
    rozptyl = maxScore - minScore

    f = open(OUTPUTOTHERFILE + "_Repl_" + str(REPLCOUNT) + "_ACTIVATION_" + ACTIVATION + "_LAYERS_COUNT_" + str(
        LAYERS_COUNT) + ".txt", "w")
    f.write("Priemerny cas trenovania :" + str(TRAINING_TIME) + '\n')
    f.write("Max skore : " + str(maxScore) + '\n')
    f.write("Min Skore : " + str(minScore) + '\n')
    f.write("Rozptyl :" + str(rozptyl) + '\n')
    f.close()

def printData(repl):
    print("Siet INFO : ")
    print("Replikacia : " + str(repl))
    print("Data")
    print(data)
    print("Training Time" + str(TRAINING_TIME))
    print("__________________")

# next_state, reward, terminated, truncated, info, action, state, done, dqn = None, None, None, None, None, None, None, None, None

if __name__ == "__main__":
    with cProfile.Profile() as profile:
    # argumenty programu : pocet vrstiev [1..3], aktivacne funkcie, csv output nazov, txt output nazov,
        OUTPUTREWFILE += sys.argv[3]
        OUTPUTOTHERFILE += sys.argv[4]
        LAYERS_COUNT = int(sys.argv[1])
        ACTIVATION =  sys.argv[2]
        for repl in range(REPLCOUNT):  # Monte Carlo
            print("Replikacia ", repl)
            timeStart = datetime.now()
            dqn = DQN(STATESIZE, ACTION_SIZE, True)
            time_step = 0
            timer2 = datetime.now()
            lives = 4
            bestFound = 0
            for i in range(C_EPISODES):
                done = False
                totalRewardPerGame = 0
                time_step = 0
                envE.reset()
                state = preprocess_state(envE.render())
                action = dqn.epsilon_greedy(state)
                for t in range(C_STEPS):
                    if t < STARTING_GAME:
                        envE.step(10)
                        continue
                    time_step += 1
                    next_state, reward, terminated, truncated, info = envE.step(action)
                    #env3.step(action)
                    #env3.render()
                    #sleep(0.01)
                    done = terminated or truncated
                    next_state = preprocess_state(next_state)
                    if done: # rozdiel oproti Q a double Q, kedy tu aj dalsiu akciu riesim
                        next_action = -1
                    else:
                        next_action = dqn.epsilon_greedy(next_state)
                    action = tf.convert_to_tensor(action)
                    done = tf.convert_to_tensor(done)
                    if reward <= 0:
                        reward -= 0.005
                    if info['lives'] < lives:
                        reward -= 50
                        lives = info['lives']
                    reward = tf.convert_to_tensor(reward)
                    dqn.store_transition(state, action, reward, next_state, next_action, done)
                    action = next_action
                    state = next_state
                    totalRewardPerGame += reward
                    #if time_step % dqn.update_rate == 0:
                    if len(dqn.replay_buffer) >= MINIBATCH_SIZE:
                        states, q1 = dqn.prepareData()
                        dqn.train(states, q1)
                    if done:
                        print('ENDED BY DESTROY ', i)
                        break

                totalRewardPerGame = totalRewardPerGame.numpy().item()
                print("Skore ziskane : ", totalRewardPerGame)

                if repl > 0:
                    data['Reward'][i] += totalRewardPerGame
                else:
                    data['Reward'].append(totalRewardPerGame)
                    data['Episode'].append(i)
                if dqn.epsilon > MIN_EPSILON:
                    dqn.epsilon *= EPSILON_DECAY
                    dqn.epsilon = max(MIN_EPSILON, dqn.epsilon)
                print(dqn.epsilon)
            print("Replikacia zabrala : ", (datetime.now() - timer2).total_seconds())
            gc.collect()
            printData(repl)
            timeEnd = datetime.now()
            TRAINING_TIME += (timeEnd - timeStart).seconds
            model1Path = "nn/SARSALEARN_Repl_" + str(repl) + "_ACTIVATION_" + ACTIVATION + "_LAYERS_COUNT_" + str(
                    LAYERS_COUNT) + "_1.h5"
            dqn.main_network1.save(model1Path)

        envE.close()
        #env3.close()
        print(data)
        for i in range(len(data['Reward'])):
            data['Reward'][i] /= REPLCOUNT
        print(data)
    #saveData(TRAINING_TIME)
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.dump_stats("resultsDlhyRun2.prof")

    dqn1 = DQN(STATESIZE, ACTION_SIZE, False)
    dqn1.main_network1 = load_model("nn/SARSALEARN_Repl_0_ACTIVATION_relu_LAYERS_COUNT_3_1.h5")
    #dqn1.main_network2 = load_model("nn/SARSA_Repl_0_ACTIVATION_relu_LAYERS_COUNT_1_2.h5")

    dqn1.epsilon = 0.0
    env2 = gym.make('Seaquest-v4', render_mode='rgb_array')
    env2.metadata['render_fps'] = 300
    env1 = HumanRendering(env2)
    env1.reset()
    #state = None
    state = preprocess_state(env2.render())
    done = False
    while not done:
        state = preprocess_state(env2.render())
        action = dqn1.epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env1.step(action)
        done = terminated or truncated
        if done:
            break
        sleep(0.05)

    env1.close()
    env2.close()
