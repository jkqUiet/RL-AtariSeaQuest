import cProfile
import random
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION'] = '0'

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

STATE_SIZE = (88, 80, 1)
#STATE_SIZE = (210, 160, 1)
ACTION_SIZE = envE.action_space.n
MINIBATCH_SIZE = 8
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.93

OUTPUTREWFILE = "rewData/"
OUTPUTOTHERFILE = "otherFiles/"
ACTIVATION = 'relu'
LAYERS_COUNT = 0

STARTING_GAME = 45

PREVACTION = 10

REPLCOUNT = 10
C_EPISODES = 40
C_STEPS = 1000

data = {
    'Reward': [],
    'Episode': [],
}

MIN_REWARD_TO_SAVE = 10
TRAINING_TIME = 0

@tf.function
def preprocess_state(image):
    image = tf.cast(image, dtype=tf.float16) / 255
    image = tf.strided_slice(image, [1, 0, 0], [176, image.shape[1], image.shape[2]], [2, 2, 1])
    image = tf.image.rgb_to_grayscale(image)
    return image

class DQN:
    def __init__(self, state_size, action_size, createNN):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=100000)
        self.gamma = 0.98
        self.epsilon = 0.9
        self.update_rate = 1000
        if createNN:
            self.main_network1 = self.build_network()
            self.target_network1 = self.build_network()
            self.target_network1.set_weights(self.main_network1.get_weights())

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(4, 4), input_shape=self.state_size, activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Flatten())
        for i in range(LAYERS_COUNT):
            model.add(Dense(48, activation=ACTIVATION))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    @tf.function
    def epsilon_greedy(self, state):
        if tf.random.uniform([], 0, 1) <= self.epsilon:
            return tf.random.uniform([], 0, self.action_size, dtype=tf.int32)
        with tf.device('/GPU:0'):
            stateT = tf.convert_to_tensor(state[None, :], dtype=tf.float16)
            Q_values1 = self.main_network1(stateT, training=False)
        return tf.argmax(Q_values1[0], output_type=tf.int32)

    @tf.function
    def prepareData(self):
        #minibatch : 0 = states, 1 = actions, 2 = rewards, 3 = next_states, 4 = dones
        transitions = list(self.replay_buffer)
        states, actions, rewards, next_states, dones = map(tf.stack, zip(*transitions))
        indices = tf.random.shuffle(tf.range(len(transitions)))[:MINIBATCH_SIZE]

        minibatch_states = tf.gather(states, indices)
        minibatch_actions = tf.gather(actions, indices)
        minibatch_rewards = tf.gather(rewards, indices)
        minibatch_next_states = tf.gather(next_states, indices)
        minibatch_dones = tf.gather(dones, indices)

        predictsNextQ = self.target_network1(minibatch_next_states, training=False)
        notDones = 1.0 - tf.cast(minibatch_dones, tf.float32)
        targetQs = minibatch_rewards + self.gamma * tf.reduce_max(predictsNextQ, axis=1) * notDones

        currentQs = self.main_network1(minibatch_states, training=False)

        indices = tf.stack([tf.range(MINIBATCH_SIZE), tf.cast(minibatch_actions, tf.int32)], axis=1)

        currentQs = tf.tensor_scatter_nd_update(currentQs, indices, targetQs)

        return minibatch_states, currentQs

    def train(self, states, cQs1):
        self.main_network1.train_on_batch(states, cQs1)

    def update_target_network(self):
        self.target_network1.set_weights(self.main_network1.get_weights())

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
    #with cProfile.Profile() as profile:
    # argumenty programu : pocet vrstiev [1..3], aktivacne funkcie, csv output nazov, txt output nazov,
    OUTPUTREWFILE += sys.argv[3]
    OUTPUTOTHERFILE += sys.argv[4]
    LAYERS_COUNT = int(sys.argv[1])
    ACTIVATION = sys.argv[2]
    for repl in range(REPLCOUNT):  # Monte Carlo
        print("Replikacia ", repl)
        timeStart = datetime.now()
        dqn = DQN(STATE_SIZE, ACTION_SIZE, True)
        time_step = 0
        timer2 = datetime.now()
        for i in range(C_EPISODES):
            done = False
            totalRewardPerGame = 0
            lives = 4
            time_step = 0
            envE.reset()
            state = preprocess_state(envE.render())
            print("Episode:", i)
            for t in range(C_STEPS):
                if t < STARTING_GAME:
                    envE.step(10)
                    continue
                time_step += 1
                action = dqn.epsilon_greedy(state)
                next_state, reward, terminated, truncated, info = envE.step(action)
                #env3.step(action)
                #sleep(0.01)
                done = terminated or truncated
                next_state = preprocess_state(next_state)
                dqn.store_transition(state, action, reward, next_state, done)
                state = next_state
                totalRewardPerGame += reward
                if time_step % dqn.update_rate == 0:
                    dqn.update_target_network()
                if len(dqn.replay_buffer) >= MINIBATCH_SIZE:
                    states, q1 = dqn.prepareData()
                    dqn.train(states, q1)
                if done:
                    print('ENDED BY DESTROY ', i)
                    break
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
        model1Path = "nn/QLEARN_Repl_" + str(repl) + "_ACTIVATION_" + ACTIVATION + "_LAYERS_COUNT_" + str(
                LAYERS_COUNT) + "_1.h5"
        #dqn.update_target_network()
        dqn.target_network1.save(model1Path)

    envE.close()
    #env3.close()
    print(data)
    for i in range(len(data['Reward'])):
        data['Reward'][i] /= REPLCOUNT
    print(data)
    saveData(TRAINING_TIME)
    #results = pstats.Stats(profile)
    #results.sort_stats(pstats.SortKey.TIME)
    #results.dump_stats("resultsDlhyRun2.prof")

    """dqn1 = DQN(STATE_SIZE, ACTION_SIZE, False)
    dqn1.main_network1 = load_model("nn/DOUBLEQLEARN_Repl_0_ACTIVATION_relu_LAYERS_COUNT_1_1.h5")  # nesmiem ho ukladat do premennej ale rovno tam kam potrebujem, co som sa docital tak je to bug
    dqn1.main_network2 = load_model("nn/DOUBLEQLEARN_Repl_0_ACTIVATION_relu_LAYERS_COUNT_1_2.h5")

    dqn1.epsilon = 0.0
    env2 = gym.make('Seaquest-v4', render_mode='rgb_array')
    env2.metadata['render_fps'] = 300
    env1 = HumanRendering(env2)
    env1.reset()
    state = None
    # state = preprocess_state(env2.render())
    done = False
    while not done:
        state = preprocess_state(env2.render())
        action = dqn1.epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env1.step(action)
        done = terminated or truncated
        if done:
            break
        # state = preprocess_state(next_state)
        sleep(0.05)

    env1.close()
    env2.close()"""
