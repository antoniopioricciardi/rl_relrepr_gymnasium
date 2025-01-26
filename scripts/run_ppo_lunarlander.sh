#!/bin/sh
#
# bash scripts/run_ppo_lunarlander.sh --run-mode ppo --env-id LunarLanderRGB --background white --gravity -10 --anchors-alpha 0
# bash scripts/run_ppo_lunarlander.sh --run-mode ppo --env-id LunarLanderRGB --background white --gravity -3 --anchors-alpha 0
# bash scripts/run_ppo_lunarlander.sh --run-mode ppo --env-id LunarLanderRGB --background red --gravity -10 --anchors-alpha 0
# bash scripts/run_ppo_lunarlander.sh --run-mode ppo --env-id LunarLanderRGB --background red --gravity -3 --anchors-alpha 0

# RELATIVE
# bash scripts/run_ppo_lunarlander.sh --run-mode ppo-rel --env-id LunarLanderRGB --background white --gravity -10 --anchors-alpha 0.999

#!/bin/bash

# usage function
usage() {
    echo "Usage: $0 [--run-mode RUN_MODE] [--env-id ENV_ID] [--background BACKGROUND] [--gravity GRAVITY] [--anchors-alpha ANCHORS_ALPHA]"
    echo "Options:"
    echo "  --run-mode     Set the run mode (default: ppo)"
    echo "  --env-id       Set the environment ID (default: CarRacing-v2)"
    echo "  --background   Set the background color (default: white)"
    echo "  --gravity     Set the gravity value (default: -10)"
    echo "  --anchors-alpha Set the alpha value for the relative representation (default: 0)"
    exit 0
}


# default values
run_mode="ppo"
env_id="LunarLanderRGB"
background="white"
gravity="-10"
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
        --background)
            background="$2"
            shift 2
            ;;
        --gravity)
            gravity="$2"
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
echo "Background: $background"
echo "Gravity: $gravity"
echo "Anchors Alpha: $anchors_alpha"
# if [ "$1" == "ppo" ]

if [ $run_mode == "ppo" ]
then
    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name " $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 1 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name " $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 2 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name " $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 3 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name " $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 4 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000
elif [ $run_mode == "ppo-rel" ]
then
    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 1 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 2 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 3 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 4 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha
elif [ $run_mode == "ppo-rel-augmented" ]
then
    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 1 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs_augmented.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 2 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs_augmented.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 3 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs_augmented.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "rel_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 4 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs_augmented.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 25000000 --anchors-alpha $anchors_alpha
elif [ $run_mode == "ppo_resnet" ]
then
    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "resnet_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 1 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000 --use-resnet

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "resnet_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 2 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000 --use-resnet

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "resnet_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 3 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000 --use-resnet

    python src/zeroshotrl/ppo_lunarlander_rgb.py --track --wandb-project-name rlrepr_ppo_lunarlander --exp-name "resnet_ $gravity"_"$background"_rgb --env-id LunarLanderRGB --seed 4 --num-envs 16 --background $background --gravity " $gravity" --stack-n 4 --total-timesteps 25000000 --use-resnet
else
    echo "Invalid argument"
fi
