import argparse

# create anchors_indices folder if it doesn't exist
# if not os.path.exists('anchor_indices'):
#     os.makedirs('anchor_indices')

# parse args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-id", type=str, default="BreakoutNoFrameskip-v4", help="Environment id"
)
parser.add_argument("--num-anchors", type=int, default=3136, help="Number of anchors")
parser.add_argument(
    "--total-num-obs", type=int, default=4000, help="Total number of observations"
)
parser.add_argument("--seed", type=int, default=1, help="Random seed")

args = parser.parse_args()

env_id = args.env_id  # 'BreakoutNoFrameSkip-v4'
num_anchors = args.num_anchors  # 3136
total_num_obs = args.total_num_obs  # 20000

filename = f"{env_id}_{num_anchors}_anchor_indices_from_{total_num_obs}.txt"

# choose 3136 random indices from 0 to 20000
import random

# set seed
random.seed(args.seed)

anchor_indices = list(range(total_num_obs))
random.shuffle(anchor_indices)
anchor_indices = anchor_indices[:num_anchors]

import time

t = time.localtime()
# save year, month, day, hour, minute
current_time = time.strftime("%Y-%m-%d_%H-%M", t)

# save anchor_indices to a txt file
text = ""
for item in anchor_indices:
    text += str(item) + "\n"
print('Saving anchor indices to file: ', f"data/anchor_indices/{filename}")
with open(f"data/anchor_indices/{filename}", "w") as f:
    f.write(text)
