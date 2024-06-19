from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, GoalBall, Ball, Box, ObstacleBall, ObstacleSquare
from minigrid.minigrid_env import MiniGridEnv

# import tuple
from typing import Tuple

class EmptyDualEnv(MiniGridEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_shape: str = "square", # ball
        goal_pos: str = "right", # left
        goal_color="green",
        item_color="red",
        wall_color: str = "grey",
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir


        self.goal_shape = goal_shape
        self.goal_pos = goal_pos
        self.goal_color = goal_color
        self.item_color = item_color
        
        self.wall_color = wall_color

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height, wall_color=self.wall_color)

        # Place a goal square
        if self.goal_pos == "right":
            if self.goal_shape == "square":
                self.put_obj(Goal(color=self.goal_color), width - 2, height - 2)
                # put box in the bottom left corner
                # self.put_obj(Box(color=self.item_color), 1, height - 2)
                self.put_obj(ObstacleSquare(color=self.item_color), 1, height - 2)
            elif self.goal_shape == "ball":
                self.put_obj(GoalBall(color=self.goal_color), width - 2, height - 2)
                # put ball in the bottom left corner
                #self.put_obj(Ball(color=self.item_color), 1, height - 2)
                self.put_obj(ObstacleBall(color=self.item_color), 1, height - 2)
        elif self.goal_pos == "left":
            if self.goal_shape == "square":
                self.put_obj(Goal(color=self.goal_color), 1, height - 2)
                # put box in the bottom right corner
                # self.put_obj(Box(color=self.item_color), width - 2, height - 2)
                self.put_obj(ObstacleSquare(color=self.item_color), width - 2, height - 2)
            elif self.goal_shape == "ball":
                self.put_obj(GoalBall(color=self.goal_color), 1, height - 2)
                # put ball in the bottom right corner
                # self.put_obj(Ball(color=self.item_color), width - 2, height - 2)
                self.put_obj(ObstacleBall(color=self.item_color), width - 2, height - 2)
        
        # # Place a goal square
        # if self.goal_pos is None:
        #     # in the bottom-right corner
        #     self.put_obj(Goal(color=self.goal_color), width - 2, height - 2)
        # else:
        #     # or at the specified position
        #     self.put_obj(Goal(color=self.goal_color), self.goal_pos[0], self.goal_pos[1])
        #     # self.put_obj(Goal(color=self.goal_color), 1,6)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs
