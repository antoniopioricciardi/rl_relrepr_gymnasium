# new
conda env create -n relative_rl python=3.9.1

## on mac
pip install -r requirements.txt

## on linux
pip install -r requirements_linux.txt











# rl_relrepr
conda env create -f relative_rl.yaml
conda activate relative_rl

pip install pip install -r relative_rl_requirements.txt




# Collect actions, generate anchors and anchor indices
sh collect_data_end_to_end_atari.sh BoxingNoFrameskip-v4 collect-actions generate-anchors-indices


relative_analysis.ipynb - read stitching results








# copy content of pydrive2fs from computer where you can access monitor to the computer where you can't. Paths are:
$CACHE_HOME/pydrive2fs/{gdrive_client_id}/default.json (unless profile is specified), where the CACHE_HOME location per platform is:

macOS	
~/Library/Caches	~/.cache	%CSIDL_LOCAL_APPDATA%

Linux (*typical)	
~/.cache

Windows
%CSIDL_LOCAL_APPDATA%


command swig failed
## ON LINUX:
apt-get remove swig
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig

If you get
ImportError: libGL.so.1: cannot open shared object file: No such file or directory

apt-get update
apt-get install ffmpeg libsm6 libxext6  -y





# Train
python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name ppo_delete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000


# RELATIVE Training
bash run_ppo_carracing_seedtests.sh --run-mode ppo-rel --env-id CarRacing-v2 --background green --car-mode standard --anchors-alpha 0.999


# rl_relrepr_gymnasium
