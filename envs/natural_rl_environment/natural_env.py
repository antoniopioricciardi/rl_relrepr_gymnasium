#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import glob
import gym
import numpy as np
from gym.utils import play

from natural_rl_environment.matting import BackgroundMattingWithColor
# from matting import BackgroundMattingWithColor
from natural_rl_environment.imgsource import (
    RandomImageSource,
    RandomColorSource,
    ColorSource,
    NoiseSource,
    RandomVideoSource,
)
# from imgsource import (
#     RandomImageSource,
#     RandomColorSource,
#     ColorSource,
#     NoiseSource,
#     RandomVideoSource,
# )
import pygame
from pygame.locals import VIDEORESIZE

class RemoveElFromTupleWrapper(gym.ObservationWrapper):
    """ The env returns a 5-tuple. Remove the fourth element from the tuple returned by the environment."""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        step_tuple = self.env.step(action)
        return step_tuple[0], step_tuple[1], step_tuple[2], step_tuple[4] # obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ReplaceBackgroundEnv(gym.ObservationWrapper):

    viewer = None

    def __init__(self, env, bg_matting, natural_source):
        """
        The source must produce a image with a shape that's compatible to
        `env.observation_space`.
        """
        super(ReplaceBackgroundEnv, self).__init__(env)
        self._bg_matting = bg_matting
        self._natural_source = natural_source

    """
    *** NOT IN THE NATURAL ENV REPO (added by me as a workaround) ***
    modified version of original observation function.
    Need to make observations tuple, because around the codebase
    it is assumed that observations are tuples, which gym functions do not always return.
    Probably a version mismatch.
    """
    def observation(self, obs):
        if len(obs) != 2:
            # add a batch dimension to the observation, creating a tuple, to keep code working
            obs = (obs, 0)
        mask = self._bg_matting.get_mask(obs)
        img = self._natural_source.get_image()
        # then only take the first element of the tuple, which is the observation
        obs=obs[0]
        obs[mask] = img[mask]
        self._last_ob = obs
        return obs

    def reset(self):
        self._natural_source.reset()
        return super(ReplaceBackgroundEnv, self).reset()

    # modified from gym/envs/atari/atari_env.py
    # This makes the monitor work
    def render(self, mode="rgb_array"):
        img = self._last_ob
        if mode == "rgb_array":
            return img
        # this elif part does not work because gym.envs.classic_control is no longer in gym
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return env.viewer.isopen




def naturalenv_wrapper(env, imgsource, color="green", resource_files=None):
    shape2d = env.observation_space.shape[:2]
    if imgsource:
        if imgsource == "color":
            if color == "green":
                src_color = [0, 255, 0]# [102, 204, 102]
            elif color == "red":
                src_color = [255, 0, 0]
            elif color == "blue":
                src_color = [0, 0, 255] # [102, 102, 204]
            elif color == "violet":
                src_color = [204, 102, 204]
            elif color == "yellow":
                src_color = [255, 255, 0] #[230, 230, 102]
            else:
                raise ValueError("Unknown color {}".format(color))
            imgsource = ColorSource(shape2d, src_color)
        elif imgsource == "randomcolor":
            imgsource = RandomColorSource(shape2d)
        elif imgsource == "noise":
            imgsource = NoiseSource(shape2d)
        else:
            files = glob.glob(os.path.expanduser(resource_files))
            assert len(files), "Pattern {} does not match any files".format(
                resource_files
            )
            if imgsource == "images":
                imgsource = RandomImageSource(shape2d, files)
            else:
                imgsource = RandomVideoSource(shape2d, files)
        bg_col = (0, 0, 0)
        if env.spec.id.startswith("Pong"):
            bg_col = (144, 72, 17)
        if env.spec.id.startswith("Boxing"):
            bg_col = (110, 156,  66)
        wrapped_env = ReplaceBackgroundEnv(
            env, BackgroundMattingWithColor(bg_col), imgsource
        )
    else:
        wrapped_env = env
    return wrapped_env


class NaturalEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env_id, imgsource="color", color="green", resource_files=None, render_mode="human", zoom=4):
        super().__init__(env_id)
        env = gym.make(env_id)
        self.wrapped_env = naturalenv_wrapper(env, imgsource, color, resource_files)
        self.render_mode = render_mode
        self.reset()
        if render_mode == "human":
            rendered = self.wrapped_env.render(mode="rgb_array")

            self.zoom = zoom
            self.video_size = [rendered.shape[1], rendered.shape[0]]
            if zoom is not None:
                self.video_size = int(self.video_size[0] * zoom), int(self.video_size[1] * zoom)
                self.screen = pygame.display.set_mode(self.video_size)
                self.clock = pygame.time.Clock()

        # pressed_keys = []
        running = True
        env_done = True

        self.render_mode = render_mode

    def step(self, action):
        if self.render_mode == "human":
            self.render()
        return self.wrapped_env.step(action)

    def reset(self):
        return self.wrapped_env.reset()

    def render(self):
        rendered = self.wrapped_env.render(mode="rgb_array")
        if self.render_mode == "human":
            img = self._display_arr(self.screen, rendered, video_size=self.video_size, transpose=True)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == VIDEORESIZE:
                    video_size = event.size
                    self.screen = pygame.display.set_mode(video_size)

            pygame.display.flip()
            self.clock.tick(90)
            return img # self.wrapped_env.render(mode=mode)

    def close(self):
        return self.wrapped_env.close()

    def seed(self, seed=None):
        return self.wrapped_env.seed(seed=seed)

    def __getattr__(self, name):
        return getattr(self.wrapped_env, name)
    
    def _display_arr(self, screen, arr, video_size, transpose):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        screen.blit(pyg_img, (0, 0))