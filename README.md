


Train

'''
PYTHONPATH=./src python src/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=1 headless=True test=False train.params.config.minibatch_size=2 +debug1=False +pythonpath="" task.env.ae_type="ae" task.env.actor_init_pos="back_to_back"
'''

evaluation(TEST). Use num_envs=1 to log videos and informtations

'''
PYTHONPATH=./src python src/phc/train.py task=HumanoidAeMcpPnn6 task.env.numEnvs=2 headless=False test=True checkpoint="runs/HumanoidAeMcpPnn6PPO/nn/HumanoidAeMcpPnn6PPO.pth"  train.params.config.minibatch_size=2 +debug1=False +pythonpath="" task.env.ae_type="ae" task.env.actor_init_pos="back_to_back"
'''

task.env.ae_type can be {ae, vae, none}
task.actor_init_pos can be {random, back_to_back}
