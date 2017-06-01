from distutils.core import setup

setup(name='DeepReinforcementLearning',
      version='0.0.1',
      description='Package containing various deep reinforcement learning algorithms',
      author='Bart Keulen',
      email='bart_keulen@hotmail.com',
      packages=['drl', 'drl.ddpg', 'drl.dqn', 'drl.exploration', 'drl.naf', 'drl.replaybuffer', 'drl.rrtexploration', 'drl.utils', 'drl.ilqg'],
      package_data={'drl': ['ilqg/assets/*.png']})