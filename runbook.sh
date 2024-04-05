PYTHONPATH=./src python src/phc/train.py task=HumanoidAeMcpPnn task.env.numEnvs=2048 train.params.config.minibatch_size=256 headless=True


PYTHONPATH=./src python src/phc/train.py \
task=HumanoidAeMcpPnn \
task.env.numEnvs=2 \
headless=False \
test=True \
checkpoint=runs/Humanoid/nn/Humanoid.pth

PYTHONPATH=./src python src/phc/train.py \
task=HumanoidAeMcpPnn3 \
task.env.numEnvs=256 \
headless=False \
test=False \
train.params.config.minibatch_size=256 \
checkpoint=good/HumanoidAeMcpPnn/002.pth

PYTHONPATH=./src python src/phc/train.py \
task=HumanoidAeMcpPnn3 \
task.env.numEnvs=2 \
headless=False \
test=True \
checkpoint=runs/HumanoidAeMcpPnn3PPO/nn/HumanoidAeMcpPnn3PPO.pth