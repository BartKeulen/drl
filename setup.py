from setuptools import setup, find_packages

setup(name='drl_tarb',
      version='0.0.1',
      description='Package containing various deep reinforcement learning algorithms',
      author='Bart Keulen',
      email='bart_keulen@hotmail.com',
      packages=find_packages(),
      package_data={'drl': ['env/assets/*.png']})