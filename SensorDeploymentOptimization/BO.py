import SensorOptimizers.BayesianOptimization as bo
import numpy as np
import pickle

ROS = True

for i in range(0, 1):
    history = bo.run(iteration = 1, epsilon = 0.5, ROS = True)

    if ROS == True:
        with open('openbox_results_single_objective_ROS/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('openbox_results_single_objective_wo_ROS/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)