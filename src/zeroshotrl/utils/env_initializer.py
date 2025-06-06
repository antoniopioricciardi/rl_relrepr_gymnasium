import gymnasium as gym
from zeroshotrl.utils.preprocess_env import (
    RepeatAction,
    RescaleObservation,
    ReshapeObservation,
    FilterFromDict,
    ColorTransformObservation
)

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


# def make_env_carracing(env, seed=0, stack=0, no_op=0, action_repeat=0, max_frames=False, episodic_life=False, clip_reward=False, idx=0, capture_video=False, run_name=''):
#     def thunk(env=env):
#         # env = gym.make(env_id)
#         # env = CarRacing(continuous=False, background='red')
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         # env = NoopResetEnv(env, noop_max=30)
#         # env = MaxAndSkipEnv(env, skip=4)
#         # env = EpisodicLifeEnv(env)
#         # if "FIRE" in env.unwrapped.get_action_meanings():
#         #     env = FireResetEnv(env)
#         # env = ClipRewardEnv(env)
#         # env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = PreprocessFrameRGB((84, 84, 3), env)  # (3, 84, 84)
#         # env = gym.wrappers.GrayScaleObservation(env)
#         if stack > 1:
#             env = gym.wrappers.FrameStack(env, stack) #(4, 3, 84, 84)
#         # env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


# TODO: rename to rgb, create a single function to instantiate envs
def make_env_atari(
    env,
    seed=0,
    rgb=True,
    stack=0,
    no_op=0,
    action_repeat=0,
    max_frames=False,
    episodic_life=False,
    clip_reward=False,
    check_fire=True,
    color_transform="standard",
    filter_dict=None,
    time_limit: int = 0,
    idx=0,
    capture_video=False,
    run_name="",
):
    def thunk(env=env):
        # print('Observation space: ', env.observation_space, 'Action space: ', env.action_space)
        # env = gym.make(env_id)
        # env = CarRacing(continuous=False, background='red')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if no_op > 0:
            env = NoopResetEnv(env, noop_max=30)
        if action_repeat > 0:
            if max_frames:
                env = MaxAndSkipEnv(env, skip=action_repeat if action_repeat > 1 else 4)
            else:
                env = RepeatAction(env, repeat=action_repeat)
        if episodic_life:
            env = EpisodicLifeEnv(env)
        if check_fire and "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        if filter_dict:
            env = FilterFromDict(env, filter_dict)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        if color_transform != "standard":
            env = ColorTransformObservation(env, color=color_transform)
        env = RescaleObservation(env, rescale_value=255.0)
        if rgb:
            #     env = PreprocessFrameRGB((84, 84, 3), env)  #
            # env = ReshapeObservation(env, (3, 96, 96)) # replace with env.observation_space.shape[1],
            
            # env = ReshapeObservation(env, (shape[2], shape[0], shape[1]))
            env = ReshapeObservation(env, (3, 84, 84))
        if not rgb:
            # env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
        # env = NormalizeFrames(env)
        # env = gym.wrappers.GrayScaleObservation(env)
        if stack > 1:
            env = gym.wrappers.FrameStack(env, stack)  # (4, 3, 84, 84)
        if time_limit > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def instantiate_env(
    env_id,
    num_envs=1,
    env_variation=None,
    env_seed=0,
    num_stack=4,
    num_no_op=30,
    action_repeat=4,
    max_frames=False,
    episodic_life=False,
    clip_reward=False,
    render_mode="rgb_array",
    image_path=None,
):
    assert env_variation is not None, 'env_variation must be specified (e.g. "green")'
    if env_id.startswith("CarRacing-v2"):
        from envs.carracing.car_racing_camera_far import CarRacing

        env = CarRacing(
            continuous=False,
            render_mode=render_mode,
            background=env_variation,
            image_path=image_path,
        )
        envs = gym.vector.SyncVectorEnv(
            [
                make_env_carracing(
                    env,
                    seed=env_seed,
                    stack=num_stack,
                    no_op=num_no_op,
                    action_repeat=action_repeat,
                    max_frames=max_frames,
                    episodic_life=episodic_life,
                    clip_reward=clip_reward,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                for i in range(num_envs)
            ]
        )
    # elif env_id == "CarRacingFast-v2":
    #     from envs.carracing.car_racing_faster import CarRacing
    #     env = CarRacing(
    #         continuous=False,
    #         render_mode=render_mode,
    #         background=env_variation,
    #         image_path=image_path,
    #     )
    #     envs = gym.vector.SyncVectorEnv(
    #         [
    #             make_env_carracing(env, seed=env_seed, stack=num_stack, no_op=num_no_op, action_repeat=action_repeat,
    #                 max_frames=max_frames, episodic_life=episodic_life, clip_reward=clip_reward, idx=i, capture_video=False, run_name="test")
    #             for i in range(num_envs)
    #         ]
    #     )
    else:
        from natural_rl_environment.natural_env import NaturalEnvWrapper

        if env_variation == "plain":
            imgsource = "plain"
            env = gym.make(env_id, render_mode=render_mode)
        else:
            imgsource = "color"
            env = NaturalEnvWrapper(
                env_id, imgsource, render_mode=render_mode, color=env_variation
            )
        envs = gym.vector.SyncVectorEnv(
            [
                make_env_atari(
                    env,
                    seed=env_seed,
                    stack=num_stack,
                    no_op=num_no_op,
                    action_repeat=action_repeat,
                    max_frames=max_frames,
                    episodic_life=episodic_life,
                    clip_reward=clip_reward,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                # make_env_atari(env, stack, env_seed, i, capture_video=False, run_name="test")
                for i in range(num_envs)
            ]
        )
    return envs


""" FOR STITCHING TESTS """


def init_carracing_env(
    car_mode="standard",
    background_color="green",
    image_path=None,
    zoom=2.7,
    cust_seed=0,
    render_md="rgb_array",
    num_envs=1,
    sync_async = "sync"
):
    if car_mode == "slow":
        from zeroshotrl.envs.carracing.car_racing_slow import CarRacing
    # elif car_mode == "no_noop":
    # from zeroshotrl.envs.carracing.car_racing_nonoop import CarRacing
    elif car_mode == "no_noop_4as":
        from zeroshotrl.envs.carracing.car_racing_nonoop_4as import CarRacing
    elif car_mode == "scrambled":
        from zeroshotrl.envs.carracing.car_racing_scrambled import CarRacing
    elif car_mode == "noleft":
        from zeroshotrl.envs.carracing.car_racing_noleft import CarRacing
    # elif car_mode == "heavy":
    #     from zeroshotrl.envs.carracing.car_racing_heavy import CarRacing
    elif car_mode == "camera_far":
        zoom = 1
        from zeroshotrl.envs.carracing.car_racing_camera_far import CarRacing
    # elif car_mode == "multicolor":
    #     from envs.carracing.car_racing_multicolor import CarRacing
    elif car_mode == "bus":
        from zeroshotrl.carl.envs.gymnasium.box2d.carl_vehicle_racing import CustomCarRacing
        from zeroshotrl.carl.envs.gymnasium.box2d.parking_garage.bus import Bus as custom_car

        # env = CustomCarRacing(vehicle_class=custom_car, continuous=False,
        #                       background=background_color, zoom=zoom, render_mode=render_md)
    elif car_mode == "tuktuk":
        from zeroshotrl.carl.envs.gymnasium.box2d.carl_vehicle_racing import CustomCarRacing
        from zeroshotrl.carl.envs.gymnasium.box2d.parking_garage.trike import TukTuk as custom_car

        # env = CustomCarRacing(vehicle_class=custom_car, continuous=False,
        #                       background=background_color, zoom=zoom, render_mode=render_md)
    elif car_mode == "street_car":
        from zeroshotrl.carl.envs.gymnasium.box2d.carl_vehicle_racing import CustomCarRacing
        from zeroshotrl.carl.envs.gymnasium.box2d.parking_garage.street_car import StreetCar as custom_car

        # env = CustomCarRacing(vehicle_class=custom_car, continuous=False,
        #                       background=background_color, zoom=zoom, render_mode=render_md)
    else:
        from zeroshotrl.envs.carracing.car_racing import CarRacing
    if car_mode in ["bus", "tuktuk", "street_car"]:
        env = CustomCarRacing(vehicle_class=custom_car, continuous=False,
                              background=background_color, zoom=zoom, render_mode=render_md)
    else:
        env = CarRacing(
            continuous=False, background=background_color, zoom=zoom, render_mode=render_md
        )      
    if sync_async == "sync":
        env_fn = gym.vector.SyncVectorEnv
    else:
        env_fn = gym.vector.AsyncVectorEnv
    # nv = gym.vector.SyncVectorEnv(
    nv = env_fn(               
        [
            make_env_atari(
                env,
                seed=cust_seed,
                rgb=True,
                stack=4,
                no_op=0,
                action_repeat=0,
                max_frames=False,
                episodic_life=False,
                clip_reward=False,
                check_fire=False,
                idx=i,
                capture_video=False,
                run_name="test",
            )
            for i in range(num_envs)
        ]
    )
    return nv


def init_env(
    env_id,
    env_info,
    background_color="green",
    image_path=None,
    zoom=2.7,
    cust_seed=0,
    render_md="human",
    num_envs=1,
    sync_async = "sync"
):
    if env_id.startswith("CarRacing-v2"):
        # separate car mode from env_id
        car_mode = env_id.split("-")[-1]
        nv = init_carracing_env(
            car_mode=car_mode,
            background_color=background_color,
            image_path=image_path,
            zoom=zoom,
            cust_seed=cust_seed,
            render_md=render_md,
            num_envs=num_envs,
            sync_async=sync_async
        )
    elif env_id.startswith("LunarLander"):
        gravity = -10
        if "-" in env_id:
            gravity = -int(env_id.split("-")[-1])
        from zeroshotrl.envs.lunarlander.lunar_lander_rgb import LunarLanderRGB
        print("Gravity:", gravity)
        env = LunarLanderRGB(render_mode=render_md, color=background_color, gravity=gravity)
        if sync_async == "sync":
            env_fn = gym.vector.SyncVectorEnv
        else:
            env_fn = gym.vector.AsyncVectorEnv
        # nv = gym.vector.SyncVectorEnv(
        nv = env_fn(
            [
                make_env_atari(
                    env,
                    seed=cust_seed,
                    rgb=True,
                    stack=4,
                    no_op=0,
                    action_repeat=0,
                    max_frames=False,
                    episodic_life=False,
                    clip_reward=False,
                    check_fire=False,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                for i in range(num_envs)

                
            ]
        )
    elif env_id.startswith("MiniWorld"):
        # lvl = env_id.split("-")[-1]
        # import miniworld
        # env = gym.make(env_id, render_mode="human")
        # from zeroshotrl.envs.miniworld.oneroom import OneRoom
        # env = OneRoom(render_mode="human", topdown=False)
        from zeroshotrl.envs.miniworld.fourrooms import FourRooms
        env = FourRooms(render_mode="human")
        if sync_async == "sync":
            env_fn = gym.vector.SyncVectorEnv
        else:
            env_fn = gym.vector.AsyncVectorEnv
        # nv = gym.vector.SyncVectorEnv(
        nv = env_fn(
            [
                make_env_atari(
                    env,
                    seed=cust_seed,
                    rgb=True,
                    stack=4,
                    no_op=0,
                    action_repeat=0,
                    max_frames=False,
                    episodic_life=False,
                    clip_reward=False,
                    check_fire=False,
                    color_transform=background_color,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                for i in range(num_envs)
            ]
        )
    elif env_id.startswith("Wolfenstein"):
        lvl = env_id.split("-")[-1]
        from wolfenstein_rl.wolfenstein_env import Wolfenstein

        use_rgb = True if env_info == "rgb" else False
        env = Wolfenstein(level=lvl, render_mode="human").env
        if sync_async == "sync":
            env_fn = gym.vector.SyncVectorEnv
        else:
            env_fn = gym.vector.AsyncVectorEnv
        # nv = gym.vector.SyncVectorEnv(
        nv = env_fn(
            [
                make_env_atari(
                    env,
                    seed=cust_seed,
                    rgb=use_rgb,
                    stack=4,
                    no_op=0,
                    action_repeat=0,
                    max_frames=False,
                    episodic_life=False,
                    clip_reward=False,
                    check_fire=False,
                    idx=i,
                    capture_video=False,
                    run_name="test",
                )
                for i in range(num_envs)
            ]
        )
    # else:
    #     from zeroshotrl.natural_rl_environment.natural_env import NaturalEnvWrapper

    #     if background_color == "plain":
    #         imgsource = "plain"
    #         env = gym.make(env_id, render_mode=render_md)
    #     else:
    #         imgsource = "color"
    #         env = NaturalEnvWrapper(
    #             env_id, imgsource, render_mode=render_md, color=background_color
    #         )
    #     nv = gym.vector.SyncVectorEnv(
    #         [
    #             make_env_atari(
    #                 env,
    #                 seed=cust_seed,
    #                 rgb=True,
    #                 stack=4,
    #                 no_op=0,
    #                 action_repeat=4,
    #                 max_frames=False,
    #                 episodic_life=True,
    #                 clip_reward=False,
    #                 idx=i,
    #                 capture_video=False,
    #                 run_name="test",
    #             )
    #             # make_env_atari(env, stack, env_seed, i, capture_video=False, run_name="test")
    #             for i in range(1)
    #         ]
    #     )
    return nv
