import SensorOptimizers.BayesianOptimization as bo
import numpy as np
import pickle

ROS = True
epsilon = 0.25

print('----- Running BO for epsilon: ', epsilon)

for i in range(0, 5):
    history = bo.run(iteration = 1000, epsilon = epsilon, ROS = True)

    if ROS == True:
        with open('openbox_results_single_objective_ROS/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('openbox_results_single_objective_wo_ROS/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)