
import numpy as np
import copy
import torch
import yaml

from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import env_configurations
from rl_games.common import experiment
from rl_games.common import tr_helpers

from rl_games.algos_torch import network_builder
from rl_games.algos_torch import model_builder
#from rl_games.algos_torch import a2c_continuous NOTE Using commonAgent from learning directory instead of this.
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
from rl_games.torch_runner import Runner

from isaacgymenvs.utils.MyA2c_continuous import MyA2c_continuous
from isaacgymenvs.utils.MyPlayer import MyPlayer

class MyRunner(Runner):
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : MyA2c_continuous(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : MyPlayer(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()

        self.algo_observer = algo_observer

        torch.backends.cudnn.benchmark = True
    def run(self, args):
        if 'checkpoint' in args and args['checkpoint'] is not None:
            if len(args['checkpoint']) > 0:
                self.load_path = args['checkpoint']

        if args['train']:
            self.run_train()
        elif args['play']:
            print('Started to play')
            player = self.create_player()
            player.restore(self.load_path)
            player.run()
        else:
            self.run_train()