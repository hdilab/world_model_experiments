import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/extract.log', level=logging.INFO)

import numpy as np
import multiprocessing as mp
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing
import cv2
import os
from hyperparameters import *
from models import vae, rnn, controller
import torch
import json
import train_rnn
import time
import sys


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def process_frame(frame):
    obs = frame[0:84, :, :]
    obs = cv2.resize(obs, (64, 64))
    return obs


def normalize_observation(observation):
    observation = cv2.resize(observation, (64, 64))
    return observation.astype('float32') / 255.


def simulate_batch(batch_num):
    og = start = time.time()
    block_print()
    with torch.no_grad():

        device = torch.device("cpu")
        vae_model = vae.ConvVAE(VAE_Z_SIZE, VAE_KL_TOLERANCE)
        if os.path.exists("checkpoints/vae_checkpoint.pth"):
            vae_model.load_state_dict(torch.load("checkpoints/vae_checkpoint.pth"))
        vae_model = vae_model.eval()
        vae_model.to(device)

        rnn_model = rnn.MDMRNN(MDN_NUM_MIXTURES, MDN_HIDDEN_SIZE, MDN_INPUT_SIZE, MDN_NUM_LAYERS, MDN_BATCH_SIZE, 1, MDN_OUTPUT_SIZE)
        if os.path.exists("checkpoints/rnn_checkpoint.pth"):
            rnn_model.load_state_dict(torch.load("checkpoints/rnn_checkpoint.pth"))
        rnn_model.to(device)
        rnn_model = rnn_model.eval()

        if os.path.exists("checkpoints.controller_checkpoint.json"):
            params = json.load("checkpoints.controller_checkpoint.json")
        else:
            cma_num_params = CMA_NUM_ACTIONS * CMA_EMBEDDING_SIZE + CMA_NUM_ACTIONS
            params = controller.get_random_model_params(cma_num_params, np.random.rand()*0.01)
        controller_model = controller.Controller(CMA_EMBEDDING_SIZE, CMA_NUM_ACTIONS, params)

        env = CarRacing()

        observations = []
        actions = []

        observation = env.reset()

        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])

        hidden_state, cell_state = train_rnn.init_hidden(MDN_NUM_LAYERS, MDN_BATCH_SIZE, MDN_HIDDEN_SIZE, device)

        observation = process_frame(observation)
        for _ in range(SEQUENCE_LENGTH + 1):
            observation = process_frame(observation)
            observations.append(observation)
            observation = normalize_observation(observation)
            observation = np.moveaxis(observation, 2, 0)
            observation = np.reshape(observation, (-1, 3, 64, 64))
            observation = torch.tensor(observation, device=device)
            mu, log_var = vae_model.encode(observation)
            embedding = vae_model.reparameterize(mu, log_var)

            controller_input = torch.cat((embedding, hidden_state.reshape(1, -1)), dim=1)
            action = controller_model.forward(controller_input)
            actions.append(action)
            observation, reward, done, info = env.step(action)
            action_tensor = torch.from_numpy(action).float().to(device)
            action_tensor = action_tensor.view(1, -1)
            rnn_inputs = torch.cat((embedding, action_tensor), dim=1)
            pi, mean, sigma, hidden_state, cell_state = rnn_model.forward(rnn_inputs, hidden_state, cell_state)

        observations = np.array(observations, dtype=np.uint8)
        actions = np.array(actions, dtype=np.float16)
        np.savez_compressed('data/obs_data_VAE_{}'.format(batch_num), obs=observations, action=actions)
        env.close()
    end = time.time()
    logging.info("_" + str(batch_num) + " Total: " + str(end-og))


def main():

    if not os.path.exists("data"):
        os.mkdir("data")
    mp.set_start_method('spawn')
    with mp.Pool(mp.cpu_count()) as p:
        p.map(simulate_batch, range(10000))

    simulate_batch(1)


if __name__ == "__main__":
    main()
