import SensorOptimizers.BayesianOptimization as bo
import numpy as np
import pickle

acquisition_function = 'kg'
acq_optimizer_type = 'auto'
ROS = True
epsilon = 0.5
error = 0.0
multi_objective = False
LSsensorsNum = 11
ISsensorsNum = 0
initial_state = 'random'
RLBO = True

sensor_types = {
    'model_motion_sensor': True,
    'model_beacon_sensor': False,
    'model_pressure_sensor': False,
    'model_accelerometer': False,
    'model_electricity_sensor': False
}


# testbed = 'Testbed2/'
# maxSensorNum = int(np.min([(5.3 / epsilon) * (8 / epsilon), LSsensorsNum]))

testbed = 'Testbed1/'
maxSensorNum = int(np.min([(8 / epsilon) * (8 / epsilon), LSsensorsNum]))

print('----- Running BO with: \n \t - epsilon: ', epsilon, 
      '\n \t - LS sensors #:', LSsensorsNum, 
      '\n \t - IS sensors #:', ISsensorsNum, 
      ' \n \t - initial state: ', initial_state)

for i in range(0, 1):
    BO = bo.BayesianOptimization(testbed = testbed,
                                 iteration = 1000, 
                                 epsilon = epsilon, 
                                 error = error,
                                 ROS = True, 
                                 LSmaxSensorNum = maxSensorNum,
                                 ISmaxSensorNum = ISsensorsNum, 
                                 initial_state = initial_state,
                                 input_sensor_types = sensor_types,
                                 acquisition_function = acquisition_function,
                                 acq_optimizer_type = acq_optimizer_type)

    history, states, actions, rewards = BO.run(RLBO = RLBO)

    with open('Results_BO/history(LS' + str(LSsensorsNum) +  'IS' + str(ISsensorsNum) + ')_' + str(i), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)