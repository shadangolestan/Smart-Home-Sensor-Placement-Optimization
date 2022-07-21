import SensorOptimizers.BinaryBayesianOptimization as bbo
import numpy as np
import pickle

ROS = True
epsilon = 1
multi_objective = True

print('----- Running BO for epsilon: ', epsilon)

for i in range(0, 5):
    history = bbo.run(iteration = 1000, epsilon = epsilon, ROS = True, multi_objective = multi_objective)

    if ROS == True:
        with open('openbox_results_single_objective_ROS/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('openbox_results_single_objective_wo_ROS/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)