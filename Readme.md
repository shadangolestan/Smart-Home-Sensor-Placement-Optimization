# Introduction
This framework performs Bayesian optimization for sensor configuration in a testbed environment. The sensor configuration is optimized in terms of maximizing the accuracy of an activity recognition. The code uses the package SensorOptimizers.BayesianOptimization which uses OpenBox.

# Requirements Installation
'pip install -r /SensorConfigurationOptimization/requirements.txt'

# Inputs
**Config.space**: the space in which to optimize the sensor configuration.
**Config.epsilon**: the minimum manhattan distance between two sensors.
**Config.LSsensorsNum**: the number of Location Sensitive sensors (e.g., motion sensors) to be placed in the space.
**Config.ISsensorsNum**: the number of Interaction Sensitive sensors (e.g., electricity sensors) to be placed in the initial state of the space.
**Config.initial_state**: the initial state of the sensor configuration.
**Config.testbed**: the testbed environment in which the sensor configuration will be used.
**Config.bo_iteration**: the number of iterations for the Bayesian optimization algorithm.
**Config.ROS**: whether to use the ROS (Reagion Of Similarity) for the testbed environment.
**Config.input_sensor_types**: the types of sensors to be used in the sensor configuration.
**Config.acquisition_function**: the acquisition function to be used in the Bayesian optimization algorithm.
**Config.acq_optimizer_type**: the type of optimizer to be used for the acquisition function.

# Outputs
A pickle file, gets stored in Results_BO, containing the history of the sensor configuration optimization, including the sensor configuration and the corresponding activity recognition acciracy at each iteration.

# Usage
To run the code, simply execute BO.ipynb. Make sure that all the necessary input parameters are specified in the Config.py file. The pickle file with the history of sensor placement optimization will be saved in the Results_BO directory.

# Baseline Usage
This work is compared with conventional methods in the literature, i.e., Genetic and Greedy Algorithms.

## Genetic Algorithm
The Genetic Algorithm uses the SensorOptimizers.GeneticAlgorithm module to run a genetic algorithm for sensor optimization. The inputs are:

**Config.space**: the space in which to optimize the sensor configuration.
**Config.epsilon**: the minimum manhattan distance between two sensors.
**Config.initSensorNum**: the initial number of sensors to be placed in the space.
**Config.maxSensorNum**: the maximum number of sensors allowed to be placed in the space.
**Config.radius**: the radius of the sensors' sensing area.
**Config.mutation_rate**: the mutation_rate of the genetic algorithm.
**Config.crossover**: the number of splits in crossover function of genetic algorithm
**Config.survival_rate**: the survival_rate of the genetic algorithm.
**Config.survival_rate**: the reproduction_rate of the genetic algorithm.
**Config.ISsensorsNum**: the number of Interaction Sensitive sensors (e.g., electricity sensors) to be placed in the initial state of the space.
**Config.initial_state**: the initial state of the sensor configuration.
**Config.testbed**: the testbed environment in which the sensor configuration will be used.
**Config.ga_iteration**: the number of iterations for the genetic algorithm.
**Config.ROS**: whether to use the ROS (Reagion Of Similarity) for the testbed environment.
**Config.input_sensor_types**: the types of sensors to be used in the sensor configuration.

To run the genetic algoritm, run GA.ipynb. A pickle file, gets stored in GA_results, containing the history of the sensor configuration optimization, including the sensor configuration and the corresponding activity recognition acciracy at each iteration.

## Greedy Algorithm
The Greedy Algorithm uses the SensorOptimizers.Greedy module to run a greedy algorithm for sensor optimization. The inputs are:

**Config.space**: the space in which to optimize the sensor configuration.
**Config.epsilon**: the minimum manhattan distance between two sensors.
**Config.greedy_iteration**: the number of iterations for the greedy algorithm.
**Config.LSsensorsNum**: the number of Location Sensitive sensors (e.g., motion sensors) to be placed in the space.
**Config.ISsensorsNum**: the number of Interaction Sensitive sensors (e.g., electricity sensors) to be placed in the initial state of the space.
**Config.initial_state**: the initial state of the sensor configuration.
**Config.input_sensor_types**: the types of sensors to be used in the sensor configuration.


