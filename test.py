import torch

# load the following torch model: "models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_1/encoder.pt" and bring it to cpu
encoder = torch.load("models/CarRacing-v2/rgb/green/ppo/relative/relu/alpha_0_999/seed_1/encoder.pt", map_location=torch.device('cpu'))

print(encoder["obs_anchors_filename"])