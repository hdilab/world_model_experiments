import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/logs_2.log', level=logging.ERROR)

import numpy as np
import tqdm
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import torch.multiprocessing as mp
import cv2

from models import vae, rnn, controller
import torch
import pickle

import train_rnn
import os
import time
import json
import sys

from hyperparameters import *

CMA_NUM_PARAMS = CMA_NUM_ACTIONS * CMA_EMBEDDING_SIZE + CMA_NUM_ACTIONS


def block_print():
    sys.stdout = open(os.devnull, 'w')


def process_frame(frame):
    obs = frame[0:84, :, :]
    obs = cv2.resize(obs, (64, 64))
    return obs


def normalize_observation(observation):
    observation = cv2.resize(observation, (64, 64))
    return observation.astype('float32') / 255.


def get_weights_bias(params):
    weights = params[:CMA_NUM_PARAMS - CMA_NUM_ACTIONS]
    bias = params[-CMA_NUM_ACTIONS:]
    weights = np.reshape(weights, [CMA_EMBEDDING_SIZE, CMA_NUM_ACTIONS])
    return weights, bias


def decide_action(vae_model, rnn_model, controller_model, observation, hidden_state, cell_state, device):
    observation = process_frame(observation)
    observation = normalize_observation(observation)
    observation = np.moveaxis(observation, 2, 0)
    observation = np.reshape(observation, (-1, 3, 64, 64))
    observation = torch.tensor(observation, device=device)
    mu, log_var = vae_model.encode(observation)
    embedding = vae_model.reparameterize(mu, log_var)

    controller_input = torch.cat((embedding, hidden_state.reshape(1, -1)), dim=1)
    action = controller_model.forward(controller_input)
    action_tensor = torch.from_numpy(action).float().to(device)
    action_tensor = action_tensor.view(1, -1)
    rnn_inputs = torch.cat((embedding, action_tensor), dim=1)
    pi, mean, sigma, hidden_state, cell_state = rnn_model.forward(rnn_inputs, hidden_state, cell_state)
    return action, hidden_state, cell_state


def play(params):
    with torch.no_grad():
        block_print()
        device = torch.device("cpu")
        vae_model = vae.ConvVAE(VAE_Z_SIZE, VAE_KL_TOLERANCE)
        if os.path.exists("checkpoints/vae_checkpoint.pth"):
            vae_model.load_state_dict(torch.load("checkpoints/vae_checkpoint.pth", map_location=device))
        vae_model = vae_model.eval()
        vae_model.to(device)

        rnn_model = rnn.MDMRNN(MDN_NUM_MIXTURES, MDN_HIDDEN_SIZE, MDN_INPUT_SIZE, MDN_NUM_LAYERS, MDN_BATCH_SIZE, 1,
                               MDN_OUTPUT_SIZE)
        if os.path.exists("checkpoints/rnn_checkpoint.pth"):
            rnn_model.load_state_dict(torch.load("checkpoints/rnn_checkpoint.pth", map_location=device))
        rnn_model.to(device)
        rnn_model = rnn_model.eval()

        controller_model = controller.Controller(CMA_EMBEDDING_SIZE, CMA_NUM_ACTIONS, params)

        env = CarRacing()
        _NUM_TRIALS = 16
        agent_reward = 0
        for trial in range(_NUM_TRIALS):
            observation = env.reset()
            # Little hack to make the Car start at random positions in the race-track
            np.random.seed(int(str(time.time()*1000000)[10:13]))
            position = np.random.randint(len(env.track))
            env.car = Car(env.world, *env.track[position][1:4])

            hidden_state, cell_state = train_rnn.init_hidden(MDN_NUM_LAYERS, MDN_BATCH_SIZE, MDN_HIDDEN_SIZE, device)

            total_reward = 0.0
            steps = 0
            while True:
                action, hidden_state, cell_state = decide_action(vae_model, rnn_model, controller_model, observation, hidden_state, cell_state, device)
                observation, r, done, info = env.step(action)
                total_reward += r
                # NB: done is not True after 1000 steps when using the hack above for
                # 	  random init of position

                steps += 1
                if steps == 999:
                    break

            # If reward is out of scale, clip it
            total_reward = np.maximum(-100, total_reward)
            agent_reward += total_reward
        env.close()
        return - (agent_reward / _NUM_TRIALS)


def train():
    logging.debug("Finished getting cuda devices")
    reward_logs = []
    best_reward = -9999999
    if os.path.exists("checkpoints/controller.pkl"):
        fo = open('checkpoints/controller.pkl', 'rb')
        es = pickle.load(fo)
        fo.close()
    else:
        es = cma.CMAEvolutionStrategy(CMA_NUM_PARAMS * [0], 0.1, {'popsize': 64})
    rewards_through_gens = []
    generation = 1
    try:
        while not es.stop():
            solutions = es.ask()
            solutions = list(solutions)
            params = []
            for i in range(len(solutions)):
                params.append((solutions[i],))

            #with mp.Pool(16) as p:
            with mp.Pool(mp.cpu_count()) as p:
                rewards = list(tqdm.tqdm(p.starmap(play, params), total=len(solutions)))
            
            es.tell(solutions, rewards)

            rewards = np.array(rewards) * (-1.)
            logging.critical("\n**************")
            logging.critical("Generation: {}".format(generation))
            logging.critical("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
            logging.critical("Avg reward: {:.3f}".format(np.mean(rewards)))
            logging.critical("**************\n")

            reward_logs.append([generation, np.min(rewards), np.mean(rewards), np.max(rewards)])
            generation += 1
            rewards_through_gens.append(rewards)
            fo = open('checkpoints/controller.pkl', 'wb')
            pickle.dump(es, fo)
            fo.close()

            with open("checkpoints/rewards.json", 'wt') as out:
                res = json.dump(reward_logs, out)

            if np.max(rewards) > best_reward:
                best_reward = np.max(rewards)
                index = np.argmax(rewards)

                fo = open('checkpoints/params.pkl', 'wb')
                pickle.dump(solutions[index], fo)
                fo.close()

    except (KeyboardInterrupt, SystemExit):
        logging.info("Manual Interrupt")
    except Exception as e:
        logging.exception(e)
    return es


if __name__ == '__main__':
    logging.debug("Inside Main Method")
    mp.set_start_method('spawn')
    torch.no_grad()
    es = train()
    np.save('best_params', es.best.get()[0])
