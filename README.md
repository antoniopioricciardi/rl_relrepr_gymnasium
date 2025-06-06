# UV GUIDE
install uv guide (for up to date version check https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)
ON MAC/LINUX:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
ON WINDOWS:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

# Install
```bash
uv sync
```

if you get 'No such file or directory: 'clang': 'clang''
```bash
apt-get -y install clang
```

pip install dvc



# ACTIVATE ENVIRONMENT
```bash
source .venv/bin/activate
```

# Training
## standard training
```bash
bash scripts/run_ppo_carracing.sh --run-mode ppo --env-id CarRacing-v2 --background green --car-mode standard
```


# dvc: copy content of pydrive2fs from computer where you can access monitor to the computer where you can't. Paths are:
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












# new env
conda create -n relative_rl python=3.9

## on mac
pip install -r requirements_mac.txt

## on linux
pip install -r requirements_linux.txt









# rl_relrepr
conda env create -f relative_rl.yaml
conda activate relative_rl

pip install pip install -r relative_rl_requirements.txt




# Collect actions, generate anchors and anchor indices
sh collect_data_end_to_end_atari.sh BoxingNoFrameskip-v4 collect-actions generate-anchors-indices


relative_analysis.ipynb - read stitching results











# Train
python ppo_carracing_discrete_rgb_relrepr_end_to_end.py --track --wandb-project-name ppo_delete --exp-name standard_green_rgb --env-id CarRacing-custom --seed 1 --num-envs 16 --background green --car-mode standard --stack-n 4 --total-timesteps 5000000


# RELATIVE Training
bash run_ppo_carracing_seedtests.sh --run-mode ppo-rel --env-id CarRacing-v2 --background green --car-mode standard --anchors-alpha 0.999


# rl_relrepr_gymnasium
