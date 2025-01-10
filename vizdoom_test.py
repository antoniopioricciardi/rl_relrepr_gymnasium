import gymnasium
from vizdoom import gymnasium_wrapper # This import will register all the environments
# import vizdoom.gymnasium_wrapper.gymnasium_env_defns as env_defns
# env = gymnasium.make("VizdoomBasic-v0", render_mode="human") # or any other environment id


# env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
    
#     print("Reward:", reward)
#     print("Observation:", observation)
#     print("Done:", done)
#     print("Info:", info)
#     print("=====================")

# env.close()


import os

from gymnasium.utils import EzPickle

# from vizdoom import scenarios_path
from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv


class CustomVizdoomScenarioEnv(VizdoomEnv, EzPickle):
    """Basic ViZDoom environments which reside in the `scenarios` directory"""

    def __init__(
        self, scenario_file, frame_skip=1, max_buttons_pressed=1, render_mode=None
    ):
        EzPickle.__init__(
            self, scenario_file, frame_skip, max_buttons_pressed, render_mode
        )
        super().__init__(
            # os.path.join(scenario_file),
            scenario_file,
            frame_skip,
            max_buttons_pressed,
            render_mode,
        )


env = CustomVizdoomScenarioEnv("vizdoom_scenarios/my_way_home.cfg", render_mode="human")

# env = VizdoomScenarioEnv("basic_test.cfg", render_mode="human")
env.reset()
done = False

while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print("Reward:", reward)
    # print("Observation:", observation)
    print("Done:", done)
    print("Info:", info)
    print("=====================")




# import os
# import vizdoom as vzd
# game = vzd.DoomGame()
# game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg")) # or any other scenario file

# game.set_window_visible(True)
# game.set_mode(vzd.Mode.ASYNC_PLAYER)
# game.init()

# game.new_episode()

# while not game.is_episode_finished():
#     s = game.get_state()
#     game.advance_action()
#     r = game.get_last_reward()
#     print("State #" + str(s.number))
#     print("Player position x:", s.game_variables[0])
#     print("Reward:", r)
#     print("=====================")

# print("Episode finished!")
# print("Total reward:", game.get_total_reward())
# print("************************")
# game.close