# License: MIT

import sys
import time
import traceback
import math
from typing import List
from collections import OrderedDict
from tqdm import tqdm
from openbox.optimizer.base import BOBase
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import get_result
from openbox.core.base import Observation

import gym
import numpy as np
import random
# from Reinforcement_Learning.ENV import AF_ENV


"""
    The objective function returns a dictionary that has --- config, constraints, objs ---.
"""

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from gym import spaces


class SMBO(BOBase):
    """
    Parameters
    ----------
    objective_function : callable
        Objective function to optimize.
    config_space : openbox.space.Space
        Configuration space.
    num_constraints : int
        Number of constraints in objective function.
    num_objs : int
        Number of objectives in objective function.
    max_runs : int
        Number of optimization iterations.
    runtime_limit : int or float, optional
        Time budget for the whole optimization process. None means no limit.
    time_limit_per_trial : int or float
        Time budget for a single evaluation trial.
    advisor_type : str
        Type of advisor to produce configuration suggestion.
        - 'default' (default): Bayesian Optimization
        - 'tpe': Tree-structured Parzen Estimator
        - 'ea': Evolutionary Algorithms
        - 'random': Random Search
        - 'mcadvisor': Bayesian Optimization with Monte Carlo Sampling
    surrogate_type : str
        Type of surrogate model in Bayesian optimization.
        - 'gp' (default): Gaussian Process. Better performance for mathematical problems.
        - 'prf': Probability Random Forest. Better performance for hyper-parameter optimization (HPO).
        - 'lightgbm': LightGBM.
    acq_type : str
        Type of acquisition function in Bayesian optimization.
        For single objective problem:
        - 'ei' (default): Expected Improvement
        - 'eips': Expected Improvement per Second
        - 'logei': Logarithm Expected Improvement
        - 'pi': Probability of Improvement
        - 'lcb': Lower Confidence Bound
        For single objective problem with constraints:
        - 'eic' (default): Expected Constrained Improvement
        For multi-objective problem:
        - 'ehvi (default)': Expected Hypervolume Improvement
        - 'mesmo': Multi-Objective Max-value Entropy Search
        - 'usemo': Multi-Objective Uncertainty-Aware Search
        - 'parego': ParEGO
        For multi-objective problem with constraints:
        - 'ehvic' (default): Expected Hypervolume Improvement with Constraints
        - 'mesmoc': Multi-Objective Max-value Entropy Search with Constraints
    acq_optimizer_type : str
        Type of optimizer to maximize acquisition function.
        - 'local_random' (default): Interleaved Local and Random Search
        - 'random_scipy': L-BFGS-B (Scipy) optimizer with random starting points
        - 'scipy_global': Differential Evolution
        - 'cma_es': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    initial_runs : int
        Number of initial iterations of optimization.
    init_strategy : str
        Strategy to generate configurations for initial iterations.
        - 'random_explore_first' (default): Random sampled configs with maximized internal minimum distance
        - 'random': Random sampling
        - 'default': Default configuration + random sampling
        - 'sobol': Sobol sequence sampling
        - 'latin_hypercube': Latin hypercube sampling
    initial_configurations : List[Configuration], optional
        If provided, the initial configurations will be evaluated in initial iterations of optimization.
    ref_point : List[float], optional
        Reference point for calculating hypervolume in multi-objective problem.
        Must be provided if using EHVI based acquisition function.
    history_bo_data : List[OrderedDict], optional
        Historical data for transfer learning.
    logging_dir : str
        Directory to save log files.
    task_id : str
        Task identifier.
    random_state : int
        Random seed for RNG.
    """
    def __init__(self, objective_function: callable, config_space,
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 runtime_limit=None,
                 time_limit_per_trial=180,
                 advisor_type='default',
                 surrogate_type='auto',
                 acq_type='auto',
                 acq_optimizer_type='auto',
                 initial_runs=3,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 advisor_kwargs: dict = None,
                 epsilon = None,
                 error = None,
                 **kwargs):
        

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, history_bo_data=history_bo_data)

        self.advisor_type = advisor_type
        advisor_kwargs = advisor_kwargs or {}

        if advisor_type == 'default':
            print('adviser type is default...')
            from openbox.core.generic_advisor import Advisor
            self.config_advisor = Advisor(config_space,
                                          num_objs=num_objs,
                                          num_constraints=num_constraints,
                                          initial_trials=initial_runs,
                                          init_strategy=init_strategy,
                                          initial_configurations=initial_configurations,
                                          optimization_strategy=sample_strategy,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          acq_optimizer_type=acq_optimizer_type,
                                          ref_point=ref_point,
                                          history_bo_data=history_bo_data,
                                          task_id=task_id,
                                          output_dir=logging_dir,
                                          random_state=random_state,
                                          epsilon = epsilon,
                                          error = error,
                                          **advisor_kwargs)
        elif advisor_type == 'mcadvisor':
            from openbox.core.mc_advisor import MCAdvisor
            self.config_advisor = MCAdvisor(config_space,
                                            num_objs=num_objs,
                                            num_constraints=num_constraints,
                                            initial_trials=initial_runs,
                                            init_strategy=init_strategy,
                                            initial_configurations=initial_configurations,
                                            optimization_strategy=sample_strategy,
                                            surrogate_type=surrogate_type,
                                            acq_type=acq_type,
                                            acq_optimizer_type=acq_optimizer_type,
                                            ref_point=ref_point,
                                            history_bo_data=history_bo_data,
                                            task_id=task_id,
                                            output_dir=logging_dir,
                                            random_state=random_state,
                                            **advisor_kwargs)
        elif advisor_type == 'tpe':
            from openbox.core.tpe_advisor import TPE_Advisor
            assert num_objs == 1 and num_constraints == 0
            self.config_advisor = TPE_Advisor(config_space, task_id=task_id, random_state=random_state,
                                              **advisor_kwargs)
        elif advisor_type == 'ea':
            from openbox.core.ea_advisor import EA_Advisor
            assert num_objs == 1 and num_constraints == 0
            self.config_advisor = EA_Advisor(config_space,
                                             num_objs=num_objs,
                                             num_constraints=num_constraints,
                                             optimization_strategy=sample_strategy,
                                             batch_size=1,
                                             task_id=task_id,
                                             output_dir=logging_dir,
                                             random_state=random_state,
                                             **advisor_kwargs)
        elif advisor_type == 'random':
            from openbox.core.random_advisor import RandomAdvisor
            self.config_advisor = RandomAdvisor(config_space,
                                                num_objs=num_objs,
                                                num_constraints=num_constraints,
                                                initial_trials=initial_runs,
                                                init_strategy=init_strategy,
                                                initial_configurations=initial_configurations,
                                                surrogate_type=surrogate_type,
                                                acq_type=acq_type,
                                                acq_optimizer_type=acq_optimizer_type,
                                                ref_point=ref_point,
                                                history_bo_data=history_bo_data,
                                                task_id=task_id,
                                                output_dir=logging_dir,
                                                random_state=random_state,
                                                **advisor_kwargs)
        else:
            raise ValueError('Invalid advisor type!')


        class AF_ENV(gym.Env):
            def __init__(self):
                from Reinforcement_Learning.ManageCONST import readCONST
                super(AF_ENV, self).__init__()
                self.CONST = readCONST()
                self.reward_range = (0, 1)
                self.States = {}
                self.action_space = spaces.Discrete(3,)
                self.observation_space = spaces.Box(low = np.array([0, 25]), high = np.array([10, 45]), dtype=np.int16)

                # self.observation_space = tuple((Box(low = np.array([0]), high = np.array([100])), 
                #                                 Box(low = np.array([0]), high = np.array([100]))))

                self.next_action = 0
                self.f_star = 20
                self.f_minus = 20
                self.s = 5
                self.buffer = 5
                self.uncertainty = 5
                # self.state = (0, self.f_star, self.f_minus)
                self.state = (self.buffer, self.uncertainty)
                

            def step(self, next_action):
                # calculate the next state:
                # The iterate function updates self.eta and self.s:
                # state = [tradeoff_buffer, f_star]

                # print('\t self.next_action:', self.next_action)
                self.next_action = next_action
                self.buffer = min(max(self.state[0] + self.next_action, 0), 10)
                # print(self.s)
                self.uncertainty = min(int(self.s * 10), 45)

                # print('\n\t ----- tradeoff_buffer: {} ----- '.format(tradeoff_buffer))

                self.state = (self.buffer, self.uncertainty)

                # print('\t ---- next state after applying buffer: ', self.state)
                # reward =  (self.f_star - self.f_minus) / self.f_star
                
                # if self.f_minus > 50:
                #     reward = -1 # * (self.f_minus / 100)

                # else:
                reward =  1 - (self.f_minus / 100)

                print('\t ----- reward: {} for f_star and f_minus: {} , {}'.format(reward, self.f_star, self.f_minus))
                print('\t ---------- State is: ', self.state)

                info = {}
                return self.state, reward, False, info

            def reset(self):
                self.state = (0, 5)
                return self.state

        self.env = AF_ENV()

    def run_RLBO(self, q_table = None, env = None, RLBO = False):
        from statistics import mean
        from statistics import stdev
        import time
        from csv import writer
        from Reinforcement_Learning.ManageCONST import readCONST

        #env = AF_ENV()
        
        CONST = readCONST()
        MAX_EPISODES = CONST["QLearning"]["MAX_EPISODES"]
        MAX_TRY = CONST["QLearning"]["MAX_TRY"]
        learning_rate = CONST["QLearning"]["learning_rate"]
        gamma = CONST["QLearning"]["gamma"]
        epsilon = CONST["QLearning"]["epsilon"]
        epsilon_decay = CONST["QLearning"]["epsilon_decay"]
        #action_num = tuple((env.action_space.high + np.ones(env.action_space.shape)).astype(int))
        #q_table = np.zeros(num_box + action_num)

        num_box = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))


        # q_table = np.zeros(num_box + (self.env.action_space.n,))
        q_table = np.random.random(num_box + (self.env.action_space.n,))


        self.env.States = {}
        state = self.env.reset()
        total_reward = 0

        # state = tuple(state)

        states = []
        actions = []
        rewards = []

        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            if self.budget_left < 0:
                self.logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = self.env.action_space.sample()
                
            else:                
                action = np.random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))


            # print('buffer to be applied in the next iteration: ', action - 1)
            
            # Do action and get result
            self.iterate(budget_left=self.budget_left, RLBO = RLBO, rl_action = action - 1)
            next_state, reward, _, _ = self.env.step(action - 1)

            total_reward += reward
            q_value = q_table[state][action]
            best_q = np.max(q_table[state])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))            
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            if epsilon >= 0.005:
                epsilon *= epsilon_decay

            runtime = time.time() - start_time
            self.budget_left -= runtime
    
            states.append(state)
            rewards.append(reward)
            actions.append(action)
    
        
        return self.get_history(), states, actions, rewards

    def run(self):
        from statistics import mean
        from statistics import stdev
        import time
        from csv import writer

        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            if self.budget_left < 0:
                self.logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()

            self.iterate(budget_left=self.budget_left)

            runtime = time.time() - start_time
            self.budget_left -= runtime
        return self.get_history()

    def iterate(self, budget_left=None, RLBO = False, rl_action = None):
        # get configuration suggestion from advisor

        # TODO: HERE I SHOULD ADD RL ACTIONS
        if not rl_action == None:
            self.env.f_star = self.config_advisor.get_f_star()
            # self.env.s = self.config_advisor.get_variance()
            config = self.config_advisor.get_suggestion(rl_action = rl_action, RLBO = RLBO)


        else:
            config = self.config_advisor.get_suggestion()

        trial_state = SUCCESS
        _budget_left = int(1e10) if budget_left is None else budget_left
        _time_limit_per_trial = math.ceil(min(self.time_limit_per_trial, _budget_left))

        # only evaluate non duplicate configuration

        if config not in self.config_advisor.history_container.configurations:
            start_time = time.time()
            try:
                # evaluate configuration on objective_function within time_limit_per_trial
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     _time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % _time_limit_per_trial)
                else:
                    # parse result
                    objs, constraints = get_result(_result)
            except Exception as e:
                # parse result of failed trial
                if isinstance(e, TimeoutException):
                    self.logger.warning(str(e))
                    trial_state = TIMEOUT
                else:
                    self.logger.warning('Exception when calling objective function: %s' % str(e))
                    trial_state = FAILED
                objs = self.FAILED_PERF
                constraints = None

            elapsed_time = time.time() - start_time
            # update observation to advisor
            observation = Observation(
                config=config, objs=objs, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time,
            )
            if _time_limit_per_trial != self.time_limit_per_trial and trial_state == TIMEOUT:
                # Timeout in the last iteration.
                pass
            else:
                self.config_advisor.update_observation(observation)
        else:
            self.logger.info('This configuration has been evaluated! Skip it: %s' % config)
            history = self.get_history()
            config_idx = history.configurations.index(config)
            trial_state = history.trial_states[config_idx]
            objs = history.perfs[config_idx]
            constraints = history.constraint_perfs[config_idx] if self.num_constraints > 0 else None
            if self.num_objs == 1:
                objs = (objs,)

        self.iteration_id += 1
        # Logging.
        if self.num_constraints > 0:
            self.logger.info('Iteration %d, objective value: %s. constraints: %s.'
                             % (self.iteration_id, objs, constraints))
        else:
            self.logger.info('Iteration %d, objective value: %s.' % (self.iteration_id, objs))
            

        # Visualization.
        # for idx, obj in enumerate(objs):
        #     if obj < self.FAILED_PERF[idx]:
        #         self.writer.add_scalar('data/objective-%d' % (idx + 1), obj, self.iteration_id)

        if not rl_action == None:
            # self.env.s = np.mean(self.config_advisor.get_variance())
            
            # self.env.f_minus = self.config_advisor.get_f_minus()


            self.env.s = np.mean(self.config_advisor.get_variance())
            # perfs = history_container.get_perfs()

            self.env.f_minus = objs[0]


            

        return config, trial_state, constraints, objs
