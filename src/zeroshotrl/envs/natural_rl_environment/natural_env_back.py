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
from gym.utils import play

from matting import BackgroundMattingWithColor
from imgsource import (
    RandomImageSource,
    RandomColorSource,
    NoiseSource,
    RandomVideoSource,
)


class RemoveElFromTupleWrapper(gym.ObservationWrapper):
    """The env returns a 5-tuple. Remove the fourth element from the tuple returned by the environment."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        step_tuple = self.env.step(action)
        return (
            step_tuple[0],
            step_tuple[1],
            step_tuple[2],
            step_tuple[4],
        )  # obs, reward, done, info

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
    modified version of original observation function.
    Need to make observations tuple, because around the codebase
    it is assumed that observations are tuples, which gym functions do not always return.
    Probably a version mismatch.
    """

    def observation(self, obs):
        if len(obs) != 2:
            # add a batch dimension to the observation, creating a tuple, to keep code working
            obs = (obs, 0)
        mask = self._bg_matting.get_mask(obs)
        img = self._natural_source.get_image()
        # then only take the first element of the tuple, which is the observation
        obs = obs[0]
        obs[mask] = img[mask]
        self._last_ob = obs
        return obs

    # def observation(self, obs):
    #     mask = self._bg_matting.get_mask(obs)
    #     img = self._natural_source.get_image()
    #     obs[mask] = img[mask]
    #     self._last_ob = obs
    #     return obs

    def reset(self):
        self._natural_source.reset()
        return super(ReplaceBackgroundEnv, self).reset()

    # modified from gym/envs/atari/atari_env.py
    # This makes the monitor work
    def render(self, mode="human"):
        img = self._last_ob
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return env.viewer.isopen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="The gym environment to base on")
    parser.add_argument("--imgsource", choices=["color", "noise", "images", "videos"])
    parser.add_argument(
        "--resource-files", help="A glob pattern to obtain images or videos"
    )
    parser.add_argument("--dump-video", help="If given, a directory to dump video")
    args = parser.parse_args()

    # the two lines below are to be deleted
    args.env = "SpaceInvadersNoFrameskip-v4"  # "EnduroNoFrameskip-v4" # "BreakoutNoFrameskip-v4"
    args.imgsource = "videos"  # "images" # "videos" # "images" # "noise"
    args.dump_video = False
    env = gym.make(args.env)
    shape2d = env.observation_space.shape[:2]

    # env = RemoveElFromTupleWrapper(env)

    if args.imgsource:
        if args.imgsource == "color":
            imgsource = RandomColorSource(shape2d)
        elif args.imgsource == "noise":
            imgsource = NoiseSource(shape2d)
        else:
            files = glob.glob(os.path.expanduser(args.resource_files))
            assert len(files), "Pattern {} does not match any files".format(
                args.resource_files
            )
            if args.imgsource == "images":
                imgsource = RandomImageSource(shape2d, files)
            else:
                imgsource = RandomVideoSource(shape2d, files)
        wrapped_env = ReplaceBackgroundEnv(
            env, BackgroundMattingWithColor((0, 0, 0)), imgsource
        )
    else:
        wrapped_env = env

    if args.dump_video:
        assert os.path.isdir(args.dump_video)
        wrapped_env = gym.wrappers.Monitor(wrapped_env, args.dump_video)
    play.play(wrapped_env, zoom=4)
