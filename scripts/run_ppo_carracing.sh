#!/bin/sh
#
# bash scripts/run_ppo_carracing.sh --run-mode ppo --env-id CarRacing-v2 --background green --car-mode standard --anchors-alpha 0
# bash scripts/run_ppo_carracing.sh --run-mode ppo-rel --env-id CarRacing-v2 --background multicolor --car-mode standard --anchors-alpha 0.999
# bash scripts/run_ppo_carracing.sh --run-mode ppo_resnet --env-id CarRacing-v2 --car-mode standard --background green
#!/bin/bash

# usage function
usage() {
    echo "Usage: $0 [--run-mode RUN_MODE] [--env-id ENV_ID] [--background BACKGROUND] [--car-mode CAR_MODE]"
    echo "Options:"
    echo "  --run-mode     Set the run mode (default: ppo)"
    echo "  --env-id       Set the environment ID (default: CarRacing-v2)"
    echo "  --background   Set the background color (default: green)"
    echo "  --car-mode     Set the car mode (default: standard)"
    echo "  --anchors-alpha Set the alpha value for the relative representation (default: 0)"
    exit 0
}


# default values
run_mode="ppo"
env_id="CarRacing-v2"
background="green"
car_mode="standard"
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
        --car-mode)
            car_mode="$2"
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
echo "Car Mode: $car_mode"
echo "Anchors Alpha: $anchors_alpha"
# if [ "$1" == "ppo" ]
if [ $run_mode == "ppo" ]
then
    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 2 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 3 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 4 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000
elif [ $run_mode == "ppo-rel" ]
then
    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 2 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 3 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "rel_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 4 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-relative --anchors-path data/anchors/"$env_id"/rgb_ppo_transitions_"$background"_obs.pkl --anchors-indices-path data/anchor_indices/"$env_id"_3136_anchor_indices_from_4000.txt --total-timesteps 5000000 --anchors-alpha $anchors_alpha
elif [ $run_mode == "ppo_resnet" ]
then
    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 2 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 3 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet

    python src/zeroshotrl/ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name rlrepr_ppo_carracing_discrete --exp-name "resnet_$car_mode"_"$background"_rgb --env-id CarRacing-custom --seed 4 --num-envs 16 --background $background --car-mode $car_mode --stack-n 4 --total-timesteps 5000000 --use-resnet
else
    echo "Invalid argument"
fi
