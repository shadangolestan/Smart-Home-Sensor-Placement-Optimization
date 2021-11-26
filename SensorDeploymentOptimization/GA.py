import SensorOptimizers.GeneticAlgorithm as ga
import numpy as np
import pickle

for i in range(1, 5):
    print('----- ', 'Running Genetic Algorithm #', i, ':')
    result, best_configuration_history = ga.run(iteration = 100, 
                                                population = 10,
                                                epsilon = 1, 
                                                initSensorNum = 15, 
                                                maxSensorNum = 15,  
                                                radius = 1, 
                                                mutation_rate = 0.005, 
                                                crossover = 2,
                                                survival_rate = 0.1, 
                                                reproduction_rate = 0.2,
                                                print_epochs = True,
                                                ROS = False
                                               )

    with open('GA_results/history' + str(i + 1), 'wb') as handle:
        pickle.dump([result, best_configuration_history], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('\t', 'Best accuracy found:', max(result[len(result) - 1])[0], ' sensors num used:', max(result[len(result) - 1])[1])