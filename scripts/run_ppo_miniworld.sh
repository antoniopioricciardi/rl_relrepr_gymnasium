#!/bin/sh
#
# bash scripts/run_ppo_miniworld.sh --run-mode ppo --env-id MiniWorld-OneRoom-v0 --background standard 
# bash scripts/run_ppo_miniworld.sh --run-mode ppo --env-id MiniWorld-OneRoom-v0 --background standard --total-timesteps 200000 --num-eval-eps 30
# bash scripts/run_ppo_miniworld.sh --run-mode ppo-rel --env-id MiniWorld-OneRoom-v0 --background standard --total-timesteps 200000 --num-eval-eps 30 --anchors-alpha 0.999

# FOUR ROOMS
# bash scripts/run_ppo_miniworld.sh --run-mode ppo --env-id FourRooms-v0 --background standard --total-timesteps 200000 --num-eval-eps 30

# usage function
usage() {
    echo "Usage: $0 [--run-mode RUN_MODE] [--env-id ENV_ID] [--background BACKGROUND] [--car-mode CAR_MODE]"
    echo "Options:"
    echo "  --run-mode     Set the run mode (default: ppo)"
    echo "  --env-id       Set the environment ID (default: MiniWorld-OneRoom-v0)"
    echo "  --background   Set the background color (default: standard)"
    echo "  --anchors-alpha Set the alpha value for the relative representation (default: 0)"
    echo "  --total-timesteps Set the total timesteps (default: 200000)"
    echo "  --num-eval-eps Set the number of evaluation episodes (default: 50)"
    exit 0
}


# default values
run_mode="ppo"
env_id="MiniWorld-OneRoom-v0"
background="standard"
anchors_alpha=0
total_timesteps=200000
num_eval_eps=50

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
        --anchors-alpha)
            anchors_alpha="$2"
            shift 2
            ;;
        --total-timesteps)
            total_timesteps="$2"
            shift 2
            ;;
        --num-eval-eps)
            num_eval_eps="$2"
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
echo "Anchors Alpha: $anchors_alpha"
echo "Total Timesteps: $total_timesteps"
echo "Number of Evaluation Episodes: $num_eval_eps"
# if [ "$1" == "ppo" ]
if [ $run_mode == "ppo" ]
then
    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "$env_id"_"$background"_rgb --env-id $env_id --seed 1 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "$env_id"_"$background"_rgb --env-id $env_id --seed 2 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "$env_id"_"$background"_rgb --env-id $env_id --seed 3 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "$env_id"_"$background"_rgb --env-id $env_id --seed 4 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3
elif [ $run_mode == "ppo-rel" ]
then
    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_$env_id"_"$background"_rgb --env-id $env_id --seed 1 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_$env_id"_"$background"_rgb --env-id $env_id --seed 2 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_$env_id"_"$background"_rgb --env-id $env_id --seed 3 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_$env_id"_"$background"_rgb --env-id $env_id --seed 4 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha
# elif [ $run_mode == "ppo-rel-single-anchors" ]
# then
#     python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_unified_$env_id"_"$background"_rgb --env-id $env_id --seed 1 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/CarRacing-v2-unified/rgb_ppo_transitions_red_green_bus_tuktuk_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha

#     python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_unified_$env_id"_"$background"_rgb --env-id $env_id --seed 2 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/CarRacing-v2-unified/rgb_ppo_transitions_red_green_bus_tuktuk_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha

#     python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_unified_$env_id"_"$background"_rgb --env-id $env_id --seed 3 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/CarRacing-v2-unified/rgb_ppo_transitions_red_green_bus_tuktuk_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha

#     python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "rel_unified_$env_id"_"$background"_rgb --env-id $env_id --seed 4 --num-envs 8 --background $background --stack-n 4 --use-relative --anchors-path data/anchors/CarRacing-v2-unified/rgb_ppo_transitions_red_green_bus_tuktuk_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --anchors-alpha $anchors_alpha
elif [ $run_mode == "ppo-resnet" ]
then
    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "resnet_$env_id"_"$background"_rgb --env-id $env_id --seed 1 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --use-resnet

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "resnet_$env_id"_"$background"_rgb --env-id $env_id --seed 2 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --use-resnet

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "resnet_$env_id"_"$background"_rgb --env-id $env_id --seed 3 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --use-resnet

    python src/zeroshotrl/ppo_miniworld.py --track --wandb-project-name rlrepr_ppo_miniworld --exp-name "resnet_$env_id"_"$background"_rgb --env-id $env_id --seed 4 --num-envs 8 --background $background --stack-n 4 --total-timesteps $total_timesteps --num-eval-eps $num_eval_eps --num-eval-envs 3 --use-resnet
else
    echo "Invalid argument"
fi
