# Deep Reinforcement Learning Repository

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cb49561a350d41c69bdc4495b8e37353)](https://www.codacy.com/app/bart_keulen/drl?utm_source=github.com&utm_medium=referral&utm_content=BartKeulen/drl&utm_campaign=badger)
[![CircleCI](https://circleci.com/gh/BartKeulen/drl.svg?style=shield)](https://circleci.com/gh/BartKeulen/drl)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/cb49561a350d41c69bdc4495b8e37353)](https://www.codacy.com/app/bart_keulen/drl?utm_source=github.com&utm_medium=referral&utm_content=BartKeulen/drl&utm_campaign=Badge_Coverage)

This project contains various deep reinforcement learning methods. The methods are implemented in Python using tensorflow.
Implemented methods:
- [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html) (in progress)

## Main code

The main code can be found in the drl folder.

## Examples

The examples folder contains some examples on using the code

## Notebooks

The notebooks folder contains notebooks for developing the modules, some contain extra information to explain the code.

## CI

CircleCI is used as continuous integration platform. After every push the tests are ran.
Pytest is used for testing and pytest-cov for coverage. 

Codacy is used as code reviewing platform. Only the master and develop branch are checked by Codacy.
The coverage reports from CircleCI are automatically uploaded to Codacy.

https://www.codacy.com/app/bart_keulen/drl/dashboard