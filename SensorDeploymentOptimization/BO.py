import SensorOptimizers.BayesianOptimization as bo
import numpy as np
import pickle

ROS = True
epsilon = 1
multi_objective = False
sensorNum = 13
maxSensorNum = int(np.min([(8 / epsilon) * (8 / epsilon), sensorNum]))

print('----- Running BO for epsilon: ', epsilon, 'and sensor #:', sensorNum)

for i in range(0, 5):
    history = bo.run(iteration = 1000, 
                     epsilon = epsilon, 
                     ROS = True, 
                     multi_objective = multi_objective, 
                     maxSensorNum = maxSensorNum
                    )

    if ROS == True:
        with open('Results_BO/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('Results_BO(without ROS)/history_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)