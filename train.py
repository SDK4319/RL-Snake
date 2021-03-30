import os
import sys 
import random 
import numpy as np 
from collections import deque 
import tensorflow as tf 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.initializers import RandomUniform 

import cv2 
from Snake import SnakeGame 





class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(42, activation='relu')
        self.fc2 = Dense(42, activation='relu')
        self.fc3 = Dense(42, activation='relu')
        self.fc4 = Dense(42, activation='relu')
        self.fc5 = Dense(21, activation='relu')
        self.fc6 = Dense(21, activation='relu')
        self.fc7 = Dense(8, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))


    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        q = self.fc_out(x)
        return q 



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size 
        self.action_size = action_size 

        self.discount_factor = 0.99 
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999999
        self.epsilon_min = 0.01
        self.batch_size = 64 
        self.train_start = 10000

        self.memory = deque(maxlen=20000)

        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(lr=self.learning_rate)

        self.update_target_model()


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables 
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
#GPU Memory Alloc Size 
    gpus = tf.config.experimental.list_physical_devices('GPU') 
    if gpus: 
        try: tf.config.experimental.set_virtual_device_configuration( 
            gpus[0], 
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
        except RuntimeError as e: 
            print(e)

    env = SnakeGame()
    state_size = len(env.get_state())
    action_size = 3 

    agent = DQNAgent(state_size, action_size)

    num_episode = 10000
    max_score = 0

    for e in range(num_episode):
        done = False 
        env.reset()
        state = env.get_state()
        state = np.reshape(state, [1, state_size])


        while not done:
            frame = env.get_frame()
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

            action = agent.get_action(state)

            next_state, reward, done, score = env.step(action)        
            next_state = np.reshape(next_state, [1, state_size])

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state 

            if done:
                agent.update_target_model()

                if score > max_score:
                    print("Score: ", score, " HighScore: ", max_score, " Update High Score!")
                    max_score = score 
                    agent.model.save_weights("./save_model/model", save_format="tf")
                print("Episode: {:3d} | Score: {:3d} | HighScore: {:3d} | Epsilon: {:.4f}".format(
                    e, score, max_score, agent.epsilon))

