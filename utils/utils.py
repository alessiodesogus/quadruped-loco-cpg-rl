# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import pandas
import json

#from stable_baselines.bench.monitor import load_results
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.callbacks import BaseCallback
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

""" utils.py - general utilities """

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model and vec_env parameters every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):# -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):# -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)

            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.training_env.save(stats_path)

            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))

        if self.n_calls % 500 == 0: # self.save_freq < 10000 and 
            # also print out path periodically for off-policy aglorithms: SAC, TD3, etc.
            print('=================================== Save path is {}'.format(self.save_path))
        return True


################################################################
## Printing
################################################################

def nicePrint(vec):
    """ Print single vector (list, tuple, or numpy array) """
    # check if vec is a numpy array
    if isinstance(vec,np.ndarray):
        np.set_printoptions(precision=3)
        print(vec)
        return
    currStr = ''
    for x in vec:
        currStr = currStr + '{: .3f} '.format(x)
    print(currStr)

def nicePrint2D(vec):
    """ Print 2D vector (list of lists, tuple of tuples, or 2D numpy array) """
    for x in vec:
        currStr = ''
        for y in x:
            currStr = currStr + '{: .3f} '.format(y)
        print(currStr)


################################################################
## Plotting
################################################################
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_EPLEN = True
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(timesteps, xaxis,yaxis=None):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    if yaxis is Y_EPLEN:
        y_var = timesteps.l.values
    return x_var, y_var


def plot_curves(xy_list, xaxis, title):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def plot_results(dirs, num_timesteps, xaxis, task_name):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    #plt.figure(1)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name+'Rewards')
    plt.ylabel("Episode Rewards")
    #plt.figure(2)
    xy_list = [ts2xy(timesteps_item, xaxis, Y_EPLEN) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name+'Ep Len')
    plt.ylabel("Episode Length")


######################################################################################
## Load progress/result files (make general so can use from stable-baselines or rllib)
######################################################################################

def load_rllib(path: str) -> pandas.DataFrame:
    """
    Load progress.csv and result.json file

    :param path: (str) the directory path containing the log file(s)
    :return: (pandas.DataFrame) the logged data
    """
    # get both csv and (old) json files
    progress_file = os.path.join(path, "progress.csv")
    result_file = os.path.join(path, "result.json")

    data_frames = []
    headers = []

    with open(progress_file, 'rt') as file_handler:
        data_frame = pandas.read_csv(file_handler, index_col=None)

        plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_mean'], label='episode_reward_mean')
        try:
            plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_max'], label='episode_reward_max')
            plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_min'], label='episode_reward_min')
        except:
            pass
        plt.legend()
        plt.title('Episode reward stats')
        plt.show()

        plt.plot(data_frame['timesteps_total'], data_frame['episode_len_mean'], label='episode_len_mean')
        plt.legend()
        plt.title('Episode length')
        plt.show()

    try: 
        with open(result_file, 'rt') as file_handler: 
            # result.json, check it out
            all_episode_lengths = []
            all_episode_rewards = []
            timestep_totals = []
            # read in data
            line = file_handler.readline()
            while line: 
                ep_data = json.loads(line)

                eplens = ep_data['hist_stats']['episode_lengths']
                eprews = ep_data['hist_stats']['episode_reward']
                # at the beginning will have simulated more than 100 episodes due to early terminations
                episodes_this_iter = min(ep_data['episodes_this_iter'],len(eplens))
                # buffer has previous 100 episodes, which have mostly already been counted, so just display new ones
                eplens = eplens[:episodes_this_iter]
                eprews = eprews[:episodes_this_iter]

                all_episode_lengths.extend(eplens)
                all_episode_rewards.extend(eprews)
                timestep_totals.extend( [ep_data['timesteps_total']]*len(eplens))
                line = file_handler.readline()

            plt.scatter(timestep_totals, all_episode_rewards, s=2)
            x, y_mean = window_func(np.array(timestep_totals), 
                                    np.array(all_episode_rewards), 
                                    EPISODES_WINDOW, 
                                    np.mean)
            plt.plot(x, y_mean, color='red')
            plt.title('Episode Rewards')
            plt.show()
            plt.scatter(timestep_totals, all_episode_lengths, s=2)
            x, y_mean = window_func(np.array(timestep_totals), 
                                    np.array(all_episode_lengths), 
                                    EPISODES_WINDOW, 
                                    np.mean)
            plt.plot(x, y_mean, color='red')
            plt.title('All Episode Lengths')
            plt.show()
            data_frames.append(data_frame)

        data_frame = pandas.concat(data_frames)
        return data_frame

    except:
        print('WARNING: ES - so different data for loading result.json')
        return None


def load_rllib_v2(path: str) -> pandas.DataFrame:
    """
    Load progress.csv and result.json file, for 1 of several 

    :param path: (str) the directory path containing the log file(s)
    :return: (pandas.DataFrame) the logged data
    """
    # get both csv and (old) json files
    progress_file = os.path.join(path, "progress.csv")
    result_file = os.path.join(path, "result.json")

    data_frames = []
    headers = []

    with open(progress_file, 'rt') as file_handler:
        data_frame = pandas.read_csv(file_handler, index_col=None)

    with open(result_file, 'rt') as file_handler: 
        # result.json, check it out
        all_episode_lengths = []
        all_episode_rewards = []
        timestep_totals = []
        # read in data
        line = file_handler.readline()
        while line: 
            ep_data = json.loads(line)

            eplens = ep_data['hist_stats']['episode_lengths']
            eprews = ep_data['hist_stats']['episode_reward']
            # at the beginning will have simulated more than 100 episodes due to early terminations
            episodes_this_iter = min(ep_data['episodes_this_iter'],len(eplens))
            # buffer has previous 100 episodes, which have mostly already been counted, so just display new ones
            eplens = eplens[:episodes_this_iter]
            eprews = eprews[:episodes_this_iter]

            all_episode_lengths.extend(eplens)
            all_episode_rewards.extend(eprews)
            timestep_totals.extend( [ep_data['timesteps_total']]*len(eplens))
            line = file_handler.readline()


        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)

    return data_frame, timestep_totals, all_episode_rewards, all_episode_lengths

