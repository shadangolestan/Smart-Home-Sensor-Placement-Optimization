import sys
from scipy.stats import norm
from numpy import argmax
import SIM_SIS_Libraries.SensorsClass
import SIM_SIS_Libraries.SIM_SIS_Simulator as sim_sis
import SIM_SIS_Libraries.ParseFunctions as pf
import itertools
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import pytz
import ast
import os
import random
import plotly
from openbox import Optimizer
import CASAS.al as al
import pickle
from openbox import sp

class Data:
    def __init__(self, sensorPositions, sensorTypes, space, epsilon):
        self.radius = 1
        self.placeHolders = sensorPositions
        self.sensorTypes = sensorTypes
        self.epsilon = epsilon
        self.space = space
        # self.SensorPlaceHolderSetup()
        self.sensitivity = {'3': 'pressure',
                            '4': 'accelerometer',
                            '5': 'electricity'}
        
    def frange(self, start, stop, step):
        steps = []
        while start <= stop:
            steps.append(start)
            start +=step
            
        return steps

    def GetSensorConfiguration(self):
        from collections import Counter
        sensorLocations = self.GetSensorLocations()
        _, rooms = pf.ParseWorld(simworldname = '')
        summaryDict = Counter(self.sensorTypes)

        # TODO: DIFFERENT SENSOR TYPE DEFINITIONS SHOULD BE ADDED HERE:
        configurationSummary = []
        
        IS_handled = False
        
        for key in summaryDict.keys():
            if (key == 1):
                configurationSummary.append(['motion sensors', summaryDict[key]])

            elif (key == 2):
                configurationSummary.append(['beacon sensors', summaryDict[key]])
                
            elif (key >= 3):
                if IS_handled == False:
                    ISCount = sum([v for k,v in summaryDict.items() if k >= 3])
                    configurationSummary.append(['IS', ISCount])           
                    IS_handled = True
                
        
        configurationDetails = []
        for index, loc in enumerate(sensorLocations):
            room = ""
            for r in rooms:
                if (loc[0] >= rooms[r][0][0] and loc[0] <= rooms[r][1][0] and loc[1] >= rooms[r][0][1] and loc[1] <= rooms[r][1][1]):
                    room = r
                    break

            if (self.sensorTypes[index] == 1):
                configurationDetails.append(tuple([loc, room, 'motion sensors']))

            elif (self.sensorTypes[index] == 2):
                configurationDetails.append(tuple([loc, room, 'beacon sensors']))
                
            elif (self.sensorTypes[index] == 3):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['3']]))
                
            elif (self.sensorTypes[index] == 4):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['4']]))
                
            elif (self.sensorTypes[index] == 5):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['5']]))

            else:
                configurationDetails.append(tuple([loc, room, 'motion sensors']))
        
        sensor_config = [[configurationSummary, [tuple(configurationDetails)]], self.radius]
        return sensor_config


    def GetSensorLocations(self):
        sensorLocations = []
        for index, sensorIndicator in enumerate(self.placeHolders):
            sensorLocations.append(self.placeHolders[index])

        return sensorLocations


class BOVariables:
    def __init__(self, Data_path, epsilon, initSensorNum, maxLSSensorNum, maxISSensorNum, radius, sampleSize, ROS):
        self.epsilon = epsilon
        self.Data_path = Data_path
        self.initSensorNum = initSensorNum
        self.maxLSSensorNum = maxLSSensorNum
        self.maxISSensorNum = maxISSensorNum
        self.radius = radius
        self.sensor_distribution, self.types, self.space, self.rooms, self.agentTraces = self.ModelsInitializations(ROS)

    def ModelsInitializations(self, ROS):
        #----- Space and agent models -----: 
        # simworldname = self.Data_path + '/Configuration Files/simulationWorld2.xml'
        simworldname = ''
        agentTraces = []
        
        if ROS:
            directory = os.fsencode(self.Data_path + 'Agent Trace Files ROS/')
        else:
            directory = os.fsencode(self.Data_path + 'Agent Trace Files/')
            
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"): 
                if ROS:
                    agentTraces.append(self.Data_path + 'Agent Trace Files ROS/' + filename)
                else:
                    agentTraces.append(self.Data_path + 'Agent Trace Files/' + filename)

        # Parsing the space model: 
        space, rooms = pf.ParseWorld(simworldname)

        xs = []
        for i in space:
          for j in i:
            xs.append(j)
        A = list(set(xs))
        A.sort()
        space = [A[-1], A[-2]]

        # User parameters 
        types, sensor_distribution = pf.GetUsersParameters()

        roomsList = []
        for room in sensor_distribution:
            roomsList.append(room)
              
        return sensor_distribution, types, space, rooms, agentTraces


def frange(start, stop, step):
    steps = []
    while start <= stop:
        steps.append(start)
        start +=step
        
    return steps

def MakeSensorCombinations(start, end, epsilon, sensorType, room):
    a1, b1 = makeBoundaries(epsilon, start[0], end[0])
    a2, b2 = makeBoundaries(epsilon, start[1], end[1])    
    Xs = frange(a1, b1, epsilon)
    Ys = frange(a2, b2, epsilon)
    
    points = list(itertools.product(list(itertools.product(Xs, Ys)), [room], [sensorType[0]])) 
    C = itertools.combinations(points, distribution[room][types.index(sensorType)])

    return C

def PreProcessor(df):
    '''
    df['motion sensors'] = df['motion sensors'].apply(lambda s: list(map(int, s)))
    try:
        df['beacon sensors'] = df['beacon sensors'].apply(lambda s: list(map(int, s)))
    except:
        pass
    
    try:
        df['IS'] = df['IS'].apply(lambda s: list(map(int, s)))
    except:
        pass

    pre_activity = ''
    save_index = 0

    for index, row in df.iterrows():
        save_index = index
        Activity = row['activity']

        if Activity != pre_activity:
            if pre_activity != '':
                df.at[index - 1, 'motion sensors'] += [0]
            else:
                df.at[index, 'motion sensors'] += [1]

            pre_activity = Activity
        else:
            df.at[index - 1, 'motion sensors'] += [1]

    
    df.at[save_index, 'motion sensors'] += [0]

    sensors = set([])

    previous_M = None
    previous_B = None
    output_file = []

    for index, row in df.iterrows():
      T = row['time']
      M = row['motion sensors']
      try:
        B = row['beacon sensors']
      except:
        pass

      Activity = row['activity']
      Activity = Activity.replace(' ', '_')
      MotionSensor_Names = []
      sensorNames = []
      MotionSensor_Message = []
      BeaconSensor_Names = []
      BeaconSensor_Message = []
      

      # time = convertTime(T)
      time = "2020-06-16 " + T + ".00"

      # Motion Sensor
      try:
          for i in range(len(M)):
            sensorNames.append(Name(i, 'M'))
            if M[i] == 1:
              if (previous_M != None):
                if (previous_M[i] == 0):
                  MotionSensor_Names.append(Name(i,'M'))
                  MotionSensor_Message.append('ON')

              else:
                MotionSensor_Names.append(Name(i,'M'))
                MotionSensor_Message.append('ON')

            if previous_M != None:
              if M[i] == 0 and previous_M[i] == 1:
                MotionSensor_Names.append(Name(i,'M'))
                MotionSensor_Message.append('OFF')

          previous_M = M
          
      except:
        pass
      # Beacon Sensor

      try:
        for i in range(len(B)):
          sensorNames.append(Name(i, 'B'))
          if B[i] != 0:
            BeaconSensor_Names.append(Name(i,'B'))
            BeaconSensor_Message.append(str(B[i]))

      except:
        pass

      for m in range(len(MotionSensor_Names)):
        output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)
        
      for b in range(len(BeaconSensor_Names)):
        output_file.append(time +' '+ BeaconSensor_Names[b] + ' ' + BeaconSensor_Names[b] + ' ' + BeaconSensor_Message[b] + ' ' + Activity)
        
      for s in sensorNames:
          sensors.add(s)

    return output_file, list(sensors)
    '''






    df['motion sensors'] = df['motion sensors'].apply(lambda s: list(map(int, s)))
    try:
        df['beacon sensors'] = df['beacon sensors'].apply(lambda s: list(map(int, s)))
    except:
        pass
    try:
        df['IS'] = df['IS'].apply(lambda s: list(map(int, s)))
    except:
        pass
    
    pre_activity = ''
    save_index = 0

    for index, row in df.iterrows():
        save_index = index
        Activity = row['activity']

        if Activity != pre_activity:
            if pre_activity != '':
                df.at[index - 1, 'motion sensors'] += [0]

            else:
                df.at[index, 'motion sensors'] += [1]

            pre_activity = Activity
        else:
            df.at[index - 1, 'motion sensors'] += [1]
            
    df.at[save_index, 'motion sensors'] += [0]

    sensors = set([])

    previous_M = None
    previous_B = None
    previous_I = None
    
    output_file = []

    for index, row in df.iterrows():
      T = row['time']
      M = row['motion sensors']
      try:
        B = row['beacon sensors']
      except:
        pass
    
      try:
        I = row['IS']
      except:
        pass

      Activity = row['activity']
      Activity = Activity.replace(' ', '_')
      MotionSensor_Names = []
      sensorNames = []
      MotionSensor_Message = []
      BeaconSensor_Names = []
      BeaconSensor_Message = []
      ISSensor_Names = []
      ISSensor_Message = []
      

      # time = convertTime(T)
      time = "2020-06-16 " + T + ".00"

        
    
      # Motion Sensor
      try:
          for i in range(len(M)):
                sensorNames.append(Name(i, 'M'))
                if M[i] == 1:
                      if (previous_M != None):
                        if (previous_M[i] == 0):
                          MotionSensor_Names.append(Name(i,'M'))
                          MotionSensor_Message.append('ON')

                      else:
                        MotionSensor_Names.append(Name(i,'M'))
                        MotionSensor_Message.append('ON')

                if previous_M != None:
                      if M[i] == 0 and previous_M[i] == 1:
                        MotionSensor_Names.append(Name(i,'M'))
                        MotionSensor_Message.append('OFF')

          previous_M = M
          
      except:
        pass
    
      try:
          for i in range(len(I)):
            sensorNames.append(Name(i, 'IS'))
            if I[i] == 1:
                  if (previous_I != None):
                    if (previous_I[i] == 0):
                      ISSensor_Names.append(Name(i,'IS'))
                      ISSensor_Message.append('ON')

                  else:
                    ISSensor_Names.append(Name(i,'IS'))
                    ISSensor_Message.append('ON')

            if previous_I != None:
                  if I[i] == 0 and previous_I[i] == 1:
                    ISSensor_Names.append(Name(i,'IS'))
                    ISSensor_Message.append('OFF')

          previous_I = I

      except:
          pass 

      '''
      # Beacon Sensor
      try:
        for i in range(len(B)):
              sensorNames.append(Name(i, 'B'))
              if B[i] > -200:
                    if (previous_B != None):
                        if (previous_B[i] == 0):
                          BeaconSensor_Names.append(Name(i,'B'))
                          if B[i] >= -50:
                              BeaconSensor_Message.append('1')
                            
                          elif B[i] < -50:
                              BeaconSensor_Message.append('0.5')


                    else:
                        BeaconSensor_Names.append(Name(i,'B'))
                        if B[i] >= -50:
                           BeaconSensor_Message.append('1')
                        
                        elif B[i] < -50:
                           BeaconSensor_Message.append('0.5')

              if previous_B != None:
                    if B[i] <= -200 and previous_B[i] > -200:
                        BeaconSensor_Names.append(Name(i,'B'))
                        BeaconSensor_Message.append('0')

        previous_M = M
        
      except:
        pass
       '''
    
    
       

      for m in range(len(MotionSensor_Names)):
        output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)
        
      ''' 
      for b in range(len(BeaconSensor_Names)):
        output_file.append(time +' '+ BeaconSensor_Names[b] + ' ' + BeaconSensor_Names[b] + ' ' + BeaconSensor_Message[b] + ' ' + Activity)
      '''
    
      for i_s in range(len(ISSensor_Names)):
        output_file.append(time +' '+ ISSensor_Names[i_s] + ' ' + ISSensor_Names[i_s] + ' ' + ISSensor_Message[i_s] + ' ' + Activity)
        
      for s in sensorNames:
          sensors.add(s)
    
    # for row in output_file:
    #     print(row)
    
    
    return output_file, list(sensors)




#returns the name of the sensor
def Name(number, typeSensor):
    if number < 10:
      return typeSensor + str(0) + str(number)
    else:
      return typeSensor + str(number)

#converts epoch time to human readable
def convertTime(posix_timestamp):
    tz = pytz.timezone('MST')
    dt = datetime.fromtimestamp(posix_timestamp, tz)
    time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return time

def MakeDataBoundaries(height = 10.5, width = 6.6, MaxLSSensors = 15):
    from collections import defaultdict, OrderedDict

    d = dict()

    for idx in range(MaxLSSensors):
            d['x' + str(idx)] = (0.5, width - 0.5)
            d['y' + str(idx)] = (0.5, height - 0.5)

    return d

def black_box_function(sample, simulateMotionSensors = True, simulateEstimotes = False, simulateIS = False, Plotting = False):       
    files = []
    all_sensors = set([])
    
    for agentTrace in BOV.agentTraces:
        df_ = sim_sis.RunSimulator(BOV.space, 
                                   BOV.rooms, 
                                   agentTrace,
                                   sample.GetSensorConfiguration(), 
                                   simulateMotionSensors, 
                                   simulateEstimotes,
                                   simulateIS,
                                   Plotting, 
                                   BOV.Data_path)
        
        dataFile, sensors = PreProcessor(df_)
        all_sensors.update(sensors)
        files.append(dataFile)
    
    all_sensors = list(all_sensors)
    f1_score = (al.leave_one_out(files, all_sensors)[0]) * 100
    
    try:
        return f1_score[0]
    
    except:
        return f1_score
    
'''    
def calculate_confusion_matrix(sample, simulateMotionSensors = True, simulateEstimotes = False, Plotting = False):       
    files = []
    all_sensors = set([])

    for agentTrace in BOV.agentTraces:
        df_ = sim_sis.RunSimulator(BOV.space, 
                                   BOV.rooms, 
                                   agentTrace,
                                   sample.GetSensorConfiguration(), 
                                   simulateMotionSensors, 
                                   simulateEstimotes, 
                                   Plotting, 
                                   BOV.Data_path)
        
        dataFile, sensors = PreProcessor(df_)
        all_sensors.update(sensors)
        files.append(dataFile)
        
    all_sensors = list(all_sensors)
    
    return al.get_confusion_matrix(files, all_sensors)
'''   

def function_to_be_optimized(config):
    sensorPositions = []
    sensorTypes = []
    sensor_xy = []
    excluded = []
    
    if (LSsensorTypesNum > 0):
        for i in range(1, CONSTANTS['max_LS_sensors'] + 1):
            sensor_xy.append(config['x' + str(i)] * CONSTANTS['epsilon'])
            sensor_xy.append(config['y' + str(i)] * CONSTANTS['epsilon'])
            sensorTypes.append(config['t' + str(i)])
            sensorPositions.append(sensor_xy)
            sensor_xy = []

    if (ISsensorTypesNum > 0):
        for i in range(1, CONSTANTS['max_IS_sensors'] + 1):
            object_location = config['object_location' + str(i)].split(',')
            sensor_xy.append(float(object_location[0]))
            sensor_xy.append(float(object_location[1]))
            sensorTypes.append(config['t_o' + str(i)])
            sensorPositions.append(sensor_xy)
            sensor_xy = []

    data = Data(sensorPositions, sensorTypes, BOV.space, CONSTANTS['epsilon'])

    
    # print(sensorTypes)
    
    return 100 - black_box_function(data, 
                                    simulateMotionSensors = sensor_types['model_motion_sensor'],
                                    simulateEstimotes = sensor_types['model_beacon_sensor'],
                                    simulateIS = (sensor_types['model_pressure_sensor'] and
                                                  sensor_types['model_accelerometer'] and
                                                  sensor_types['model_electricity_sensor'])
                                   )

def BuildConfigurationSearchSpace(initial_state):
    list_of_variables = []
    if (LSsensorTypesNum > 0):
        for i in range(1, CONSTANTS['max_LS_sensors'] + 1):
            if initial_state == 'fixed':
                x = sp.Int("x" + str(i), 1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon']), default_value=1)
                y = sp.Int("y" + str(i), 1, int((CONSTANTS['height'] - 1) / CONSTANTS['epsilon']), default_value=1)

                if LSsensorTypesNum > 1:
                    t = sp.Int("t" + str(i), 1, LSsensorTypesNum, default_value=random.randint(1, LSsensorTypesNum))

                else:
                    t = sp.Constant("t" + str(i), 1)

            elif(initial_state == 'random'):
                x = sp.Int("x" + str(i), 1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon']), 
                           default_value=random.randint(1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon'])))

                y = sp.Int("y" + str(i), 1, int((CONSTANTS['height'] - 1) / CONSTANTS['epsilon']), 
                           default_value=random.randint(1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon'])))

                if LSsensorTypesNum > 1:
                    t = sp.Int("t" + str(i), 1, LSsensorTypesNum, default_value=random.randint(1, LSsensorTypesNum))

                else:
                    t = sp.Constant("t" + str(i), 1)

            else:
                raise NotImplementedError (initial_state + " is not implemented yet! Try using 'fixed' or 'random' values istead")

            list_of_variables.append(x)
            list_of_variables.append(y)
            list_of_variables.append(t)
    
    if (ISsensorTypesNum > 0):
        for i in range(1, CONSTANTS['max_IS_sensors'] + 1):
            if initial_state == 'fixed':
                x_o = sp.Int("x_o" + str(i), 1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon']), default_value=1)
                y_o = sp.Int("y_o" + str(i), 1, int((CONSTANTS['height'] - 1) / CONSTANTS['epsilon']), default_value=1)
                t_o = sp.Int("t_o" + str(i), LSsensorTypesNum + 1, ISsensorTypesNum + LSsensorTypesNum, 
                           default_value=random.randint(LSsensorTypesNum + 1, ISsensorTypesNum + LSsensorTypesNum))

            elif(initial_state == 'random'):

                #TODO:
                objects = ['0.5, 2.7', '3.5, 2.7', '6.7, 1.4', '4.2, 3.2', '1.7, 6.0', '6.0, 3.6', '7.4, 3.6', '1.0, 5.5', '6.8, 5.5', '0.5, 7.1', '2.2, 7.1', '7.1, 6.8']

                objects_location = sp.Categorical('object_location' + str(i), choices = objects, default_value = random.choice(objects))

                t_o = sp.Int("t_o" + str(i), 2 + 1, 2 + ISsensorTypesNum, 
                             default_value=random.randint(2 + 1, 2 + ISsensorTypesNum))
                
                # t_o = sp.Constant("t_o" + str(i), 3)
                
            else:
                raise NotImplementedError (initial_state + " is not implemented yet! Try using 'fixed' or 'random' values istead")

            list_of_variables.append(objects_location)
            list_of_variables.append(t_o)
    
    return list_of_variables

    
def run(surrogate_type = 'prf',
        # acq_optimizer_type = 'random_scipy',
        acq_optimizer_type = 'local_random',
        acquisition_function = 'ei',
        task_id = 'SPO',
        run_on_colab = False, 
        iteration = 1000, 
        epsilon = 1, # The distance between two nodes in the space grid:
        error = 0.25,
        LSmaxSensorNum = 15,  # max location sensitive sensor numbers
        ISmaxSensorNum = 10,  # max location sensitive sensor numbers
        radius = 1, # radius of the motion sensors
        print_epochs = True,
        height = 8.0,
        width = 8.0,
        ROS = False,
        multi_objective = False,
        initial_state = 'fixed',
        input_sensor_types = {'model_motion_sensor': True, 
                              'model_beacon_sensor': False,
                              'model_pressure_sensor': False,
                              'model_accelerometer': False,
                              'model_electricity_sensor': False},
      ):

    global multi_objective_flag
    global CONSTANTS
    global runningOnGoogleColab
    global sensor_types
    global LSsensorTypesNum
    global ISsensorTypesNum
    
    runningOnGoogleColab = run_on_colab
    multi_objective_flag = multi_objective
    CONSTANTS = {
        'iterations': iteration,
        'initial_samples': 10,
        'epsilon': epsilon,
        'radius': radius,
        'height': height,
        'width': width,
        'max_LS_sensors': LSmaxSensorNum,
        'max_IS_sensors': ISmaxSensorNum,
        'error': error
    }
    
    sensor_types = input_sensor_types
    LSsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[0:2] if condition)
    ISsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[2:5] if condition)

    if (runningOnGoogleColab == True):
        from google.colab import drive    
        drive.mount('/content/gdrive', force_remount=True)
        Data_path = 'gdrive/My Drive/PhD/Thesis/Ideas/Codes/SensorDeploymentOptimization/'
        sys.path.append('gdrive/My Drive/PhD/Thesis/Ideas/Codes/SensorDeploymentOptimization/')

    else:
        Data_path = '../SensorDeploymentOptimization/'
        sys.path.append('..')

    finalResults = []
    w = CONSTANTS['width'] - 0.5
    h = CONSTANTS['height'] - 0.5

    dataBoundaries = MakeDataBoundaries(
                                        height = CONSTANTS['height'], 
                                        width = CONSTANTS['width'], 
                                        MaxLSSensors = CONSTANTS['max_LS_sensors']
                                       )
    global BOV
    BOV =  BOVariables(
                       Data_path, 
                       CONSTANTS['epsilon'], 
                       CONSTANTS['initial_samples'],
                       CONSTANTS['max_LS_sensors'], 
                       CONSTANTS['max_IS_sensors'], 
                       CONSTANTS['radius'],
                       CONSTANTS['initial_samples'],
                       ROS = True
                      )
    
    # Define Search Space
    space = sp.Space()

    
    print(CONSTANTS['width'])
    print(CONSTANTS['height'])
    
    if (multi_objective_flag == False):
        list_of_variables = BuildConfigurationSearchSpace(initial_state)

        space.add_variables(list_of_variables)
        history_list = []

        opt = Optimizer(
            function_to_be_optimized,
            space,
            max_runs = CONSTANTS['iterations'],
            surrogate_type = surrogate_type,
            acq_optimizer_type = acq_optimizer_type,
            acq_type = acquisition_function,
            time_limit_per_trial=31000,
            task_id = task_id,
            epsilon = CONSTANTS['epsilon'],
            error = CONSTANTS['error']
        )
        history = opt.run()
            
    
    return history