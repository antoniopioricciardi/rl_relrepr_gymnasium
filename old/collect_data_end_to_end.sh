# EXAMPLE:
# sh collect_data.sh --env-id CarRacing-v2 --num-steps 4000 --collect-actions --generate-anchors-indices --backgrounds green red blue
# sh collect_data.sh --env-id CarRacing-v2 --num-steps 4000 --backgrounds multicolor


# Set default values for arguments
env_id="CarRacing-v2"
num_steps=4000
collect_actions=false
generate_anchors_indices=false

# Usage function
usage() {
    echo "Usage: $0 [--env-id ENV_ID] [--num-steps NUM_STEPS] [--actions-path ACTIONS_PATH] [--collect-actions] [--generate-anchors-indices]"
    echo "Options:"
    echo "  --env-id           Set the environment ID (default: CarRacing-v2)"
    echo "  --num-steps        Set the number of steps (default: 4000)"
    echo "  --actions-path     Set the path to the actions file (default: data/actions_lists/ENV_ID_actions_NUM_STEPS.pkl)"
    echo "  --collect-actions  Collect actions list"
    echo "  --generate-anchors-indices  Generate anchor indices"
    exit 1
}

# if argument is usage, print usage and exit
if [ "$1" = "--usage" ]; then
    usage
fi

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-id)
            env_id="$2"
            shift 2
            ;;
        --num-steps)
            num_steps="$2"
            shift 2
            ;;
        --actions-path)
            actions_path="$2"
            shift 2
            ;;
        --collect-actions)
            collect_actions=true
            shift 1
            ;;
        --generate-anchors-indices)
            generate_anchors_indices=true
            shift 1
            ;;
        *)
            echo "Invalid argument: $2"
            exit 1
            ;;
    esac
done

actions_path="data/actions_lists/${env_id}_actions_${num_steps}.pkl"
# print env_id, num_steps, actions_path, collect_actions
echo "env_id: $env_id"
echo "num_steps: $num_steps"
echo "actions_path: $actions_path"
echo "collect_actions: $collect_actions"
echo "generate_anchors_indices: $generate_anchors_indices"

# Create necessary folders if they don't exist
if [ ! -d "data/anchors" ]; then
    echo "anchors folder does not exist, creating it"
    mkdir -p data/anchors
fi

if [ ! -d "data/actions_lists" ]; then
    echo "actions_lists folder does not exist, creating it"
    mkdir -p data/actions_lists
fi

if [ ! -d "data/anchor_indices" ]; then
    echo "anchor_indices folder does not exist, creating it"
    mkdir -p data/anchor_indices
fi


# Collect actions list if --collect-actions argument is passed
if [ "$collect_actions" = true ]; then
    echo "collecting actions list"
    python collect_actions.py --env-id "$env_id" --seed 1 --model-seed 1 --num-steps "$num_steps" --background green --render-mode rgb_array
fi


# Collect observations and move the anchors data to the anchors folder
echo -e "\ncollecting observations for ${env_id}_rgb_ppo_transitions_green_obs.pkl\n"
python collect_anchors_obs_ppo_end_to_end.py --env-id "$env_id" --background green --algo ppo --render-mode rgb_array --actions-path "$actions_path" --seed 1

echo -e "\ncollecting observations for ${env_id}_rgb_ppo_transitions_red_obs.pkl\n"
python collect_anchors_obs_ppo_end_to_end.py --env-id "$env_id" --background red --algo ppo --render-mode rgb_array --actions-path "$actions_path" --seed 1

echo -e "\ncollecting observations for ${env_id}_rgb_ppo_transitions_blue_obs.pkl\n"
python collect_anchors_obs_ppo_end_to_end.py --env-id "$env_id" --background blue --algo ppo --render-mode rgb_array --actions-path "$actions_path" --seed 1
# if env_id does not start with CarRacing, then generate "plain" observations instead of "yellow"
if [[ "$env_id" != CarRacing* ]]; then
    echo -e "\ncollecting observations for ${env_id}_rgb_ppo_rgb_transitions_plain_obs.pkl\n"
    python collect_anchors_obs_ppo_end_to_end.py --env-id "$env_id" --background plain --algo ppo --render-mode rgb_array --actions-path "$actions_path" --seed 1
else
    echo -e "\ncollecting observations for ${env_id}_rgb_ppo_rgb_transitions_yellow_obs.pkl\n"
    python collect_anchors_obs_ppo_end_to_end.py --env-id "$env_id" --background yellow --algo ppo --render-mode rgb_array --actions-path "$actions_path" --seed 1
fi


# echo -e "\ncollecting observations for ${env_id}_rgb_ppo_transitions_yellow_obs.pkl\n"
# python collect_anchors_obs_ppo_end_to_end.py --env-id "$env_id" --background yellow --algo ppo --render-mode rgb_array --actions-path "$actions_path" --seed 1

# Generate anchor indices if --generate-anchors-indices argument is passed
if [ "$generate_anchors_indices" = true ]; then
    echo "generating anchor indices"
    python generate_anchors_indices.py --env-id "$env_id" --num-anchors 3136 --total-num-obs "$num_steps"
fi

echo "DONE!"
# Usage:
# ./collect_data_end_to_end.sh [--env-id ENV_ID] [--num-steps NUM_STEPS] [--actions-path ACTIONS_PATH] [--collect-actions] [--generate-anchors-indices]
#
# Options:
#   --env-id           Set the environment ID (default: CarRacing-v2)
#   --num-steps        Set the number of steps (default: 4000)
#   --actions-path     Set the path to the actions file (default: data/actions_lists/ENV_ID_actions_NUM_STEPS.pkl)
#   --collect-actions  Collect actions list
#   --generate-anchors-indices  Generate anchor indices


# example usage:
# sh collect_data_end_to_end.sh --env-id PongNoFrameskip-v4 --num-steps 12000 --collect-actions --generate-anchor-indices

# provide me an example with PongNoFrameskip-v4, 12000 steps, data/actions_lists/PongNoFrameskip-v4_actions_12000.pkl, collect-actions
# sh collect_data_end_to_end.sh --env-id PongNoFrameskip-v4 --num-steps 12000 --actions-path data/actions_lists/PongNoFrameskip-v4_actions_12000.pkl