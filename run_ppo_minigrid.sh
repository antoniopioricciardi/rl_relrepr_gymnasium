#!/bin/sh
#
# bash run_ppo_minigrid.sh --run-mode ppo --env-id minigrid --grid-size 6 --goal-shape square --goal-pos right --goal-color green --item-color red --wall-color grey --anchors-alpha 0

#!/bin/bash

# usage function
usage() {
    echo "Usage: $0 [--run-mode RUN_MODE] [--env-id ENV_ID] [--background BACKGROUND] [--car-mode CAR_MODE]"
    echo "Options:"
    echo "  --run-mode     Set the run mode (default: ppo)"
    echo "  --env-id       Set the environment ID (default: minigrid)"
    echo "  --grid-size    Set the grid size (default: 6)"
    echo "  --goal-shape   Set the goal shape (default: square)"
    echo "  --goal-pos     Set the goal position (default: right)"
    echo "  --goal-color   Set the goal color (default: green)"
    echo "  --item-color   Set the item color (default: red)"
    echo "  --wall-color   Set the wall color (default: grey)"
    echo "  --anchors-alpha Set the alpha value for the relative representation (default: 0)"
    exit 0
}

# default values
run_mode="ppo"
env_id="CarRacing-v2"
grid_size=6
goal_shape="square"
goal_pos="right"
goal_color="green"
item_color="red"
wall_color="grey"
anchors_alpha=0

# parse command line arguments
# if argument is usage, print usage and exit
if [ "$1" = "--usage" ]; then
    usage
fi

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-mode)
            run_mode="$2"
            shift 2
            ;;
        --env-id)
            env_id="$2"
            shift 2
            ;;
        --grid-size)
            grid_size="$2"
            shift 2
            ;;
        --goal-shape)
            goal_shape="$2"
            shift 2
            ;;
        --goal-pos)
            goal_pos="$2"
            shift 2
            ;;
        --goal-color)
            goal_color="$2"
            shift 2
            ;;
        --item-color)
            item_color="$2"
            shift 2
            ;;
        --wall-color)
            wall_color="$2"
            shift 2
            ;;
        --anchors-alpha)
            anchors_alpha="$2"
            shift 2
            ;;
        *)
            echo "Invalid argument: $2"
            exit 1
            ;;
    esac
done

# print the values
echo "Run Mode: $run_mode"
echo "Environment ID: $env_id"
echo "Grid Size: $grid_size"
echo "Goal Shape: $goal_shape"
echo "Goal Position: $goal_pos"
echo "Goal Color: $goal_color"
echo "Item Color: $item_color"
echo "Wall Color: $wall_color"
echo "Anchors Alpha: $anchors_alpha"
# if [ "$1" == "ppo" ]
if [ $run_mode == "ppo" ]
then
    python ppo_minigrid.py --track --wandb-project-name rlrepr_ppo_minigrid --exp-name "$car_mode"_"$background"_rgb --env-id minigrid-custom --seed 1 --num-envs 16 --grid-size $grid_size --goal-shape $goal_shape --goal-pos $goal_pos --goal-color $goal_color --item-color $item_color --wall-color $wall_color --stack-n 4 --total-timesteps 500000 --num-eval-eps 20 --learning-rate 0.0001

    python ppo_minigrid.py --track --wandb-project-name rlrepr_ppo_minigrid --exp-name "$car_mode"_"$background"_rgb --env-id minigrid-custom --seed 2 --num-envs 16 --grid-size $grid_size --goal-shape $goal_shape --goal-pos $goal_pos --goal-color $goal_color --item-color $item_color --wall-color $wall_color --stack-n 4 --total-timesteps 500000 --num-eval-eps 20 --learning-rate 0.0001

    python ppo_minigrid.py --track --wandb-project-name rlrepr_ppo_minigrid --exp-name "$car_mode"_"$background"_rgb --env-id minigrid-custom --seed 3 --num-envs 16 --grid-size $grid_size --goal-shape $goal_shape --goal-pos $goal_pos --goal-color $goal_color --item-color $item_color --wall-color $wall_color --stack-n 4 --total-timesteps 500000 --num-eval-eps 20 --learning-rate 0.0001

    python ppo_minigrid.py --track --wandb-project-name rlrepr_ppo_minigrid --exp-name "$car_mode"_"$background"_rgb --env-id minigrid-custom --seed 4 --num-envs 16 --grid-size $grid_size --goal-shape $goal_shape --goal-pos $goal_pos --goal-color $goal_color --item-color $item_color --wall-color $wall_color --stack-n 4 --total-timesteps 500000 --num-eval-eps 20 --learning-rate 0.0001
# elif [ $run_mode == "ppo-rel" ]
# then
#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha
    
#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 2 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha

#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 3 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha

#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 4 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha
# elif [ $run_mode == "ppo_resnet" ]
# then
#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet

#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 2 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet

#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 3 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet

#     python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 4 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet
else
    echo "Invalid argument"
fi