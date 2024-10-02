#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gym

from natural_env import NaturalEnvWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="The gym environment to base on", default="BoxingNoFrameskip-v4"
    )
    parser.add_argument(
        "--imgsource", choices=["plain", "color", "noise", "images", "videos"]
    )
    parser.add_argument("--color", choices=["green", "red", "blue", "yellow"])
    parser.add_argument(
        "--resource-files", help="A glob pattern to obtain images or videos"
    )
    parser.add_argument("--dump-video", help="If given, a directory to dump video")
    args = parser.parse_args()

    # the two lines below are to be deleted
    # args.env = "SpaceInvadersNoFrameskip-v4" # "EnduroNoFrameskip-v4" # "BreakoutNoFrameskip-v4"
    # args.imgsource = "videos" # "images" # "videos" # "images" # "noise"
    # args.dump_video = False
    if not args.imgsource == "plain":
        env = NaturalEnvWrapper(
            args.env, args.imgsource, render_mode="human", color=args.color
        )

    else:
        env = gym.make(args.env, render_mode="human")
    score = 0
    obs = env.reset()
    for i in range(1000):
        if i == 0:
            env.step(1)  # fire
        new_obs, reward, done, _ = env.step(env.action_space.sample())
        # env.render()
        score += reward
        obs = new_obs
        if done:
            obs = env.reset()
        # wrapped_env.render()#(mode="human")
        # wrapped_env.render()
    print(score)
    # pygame.quit()
