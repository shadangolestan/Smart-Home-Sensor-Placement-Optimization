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

class Data:
    def __init__(self, sensorPositions, sensorTypes, space, epsilon):
        self.radius = 1
        self.placeHolders = sensorPositions
        self.sensorTypes = sensorTypes
        self.epsilon = epsilon
        self.space = space
        # self.SensorPlaceHolderSetup()
        
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
        for key in summaryDict:
            if (key == 1):
                configurationSummary.append(['motion sensors', summaryDict[key]])

            elif (key == 2):
                configurationSummary.append(['beacon sensors', summaryDict[key]])

                
        
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

            else:
                configurationDetails.append(tuple([loc, room, 'motion sensors']))
        
        return [[configurationSummary, [tuple(configurationDetails)]], self.radius]


    def GetSensorLocations(self):
        sensorLocations = []
        for index, sensorIndicator in enumerate(self.placeHolders):
            sensorLocations.append(self.placeHolders[index])

        return sensorLocations


class BOVariables:
    def __init__(self, Data_path, epsilon, initSensorNum, maxSensorNum, radius, sampleSize, ROS):
        self.epsilon = epsilon
        self.Data_path = Data_path
        self.initSensorNum = initSensorNum
        self.maxSensorNum = maxSensorNum
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
    df['motion sensors'] = df['motion sensors'].apply(lambda s: list(map(int, s)))
    try:
        df['beacon sensors'] = df['beacon sensors'].apply(lambda s: list(map(int, s)))
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

def MakeDataBoundaries(height = 10.5, width = 6.6, MaxSensors = 15):
    from collections import defaultdict, OrderedDict

    d = dict()

    for idx in range(MaxSensors):
            d['x' + str(idx)] = (0.5, width - 0.5)
            d['y' + str(idx)] = (0.5, height - 0.5)

    return d

def black_box_function(sample, simulateMotionSensors = True, simulateEstimotes = False, Plotting = False):       
    files = []
    all_sensors = set([])
    
    for agentTrace in BOV.agentTraces:
        # print('-' * 50)
        # print('------- RunSimulator')
        df_ = sim_sis.RunSimulator(BOV.space, 
                                   BOV.rooms, 
                                   agentTrace,
                                   sample.GetSensorConfiguration(), 
                                   simulateMotionSensors, 
                                   simulateEstimotes, 
                                   Plotting, 
                                   BOV.Data_path)
        
        # print('------- RunSimulator ended') 
        # print('------- PreProcessor')
        dataFile, sensors = PreProcessor(df_)
        # print('------- PreProcessor ended')
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

    for i in range(1, CONSTANTS['max_sensors'] + 1):
        sensor_xy.append(config['x' + str(i)] * CONSTANTS['epsilon'])
        sensor_xy.append(config['y' + str(i)] * CONSTANTS['epsilon'])
        sensorTypes.append(config['t' + str(i)])
        sensorPositions.append(sensor_xy)
        sensor_xy = []
        
        # sensorTypes.append(1)
        
    print(sensorTypes)
    data = Data(sensorPositions, sensorTypes, BOV.space, CONSTANTS['epsilon'])

    return 100 - black_box_function(data, 
                                    simulateMotionSensors = sensor_types['model_motion_sensor'],
                                    simulateEstimotes = sensor_types['model_beacon_sensor']
                                   )
'''
def confusion_matrix(config):
    sensorPositions = []
    sensor_xy = []
    excluded = []

    for i in range(1, len(config.keys()) + 1):
        try:
            sensor_xy.append(config['x' + str(i)] * CONSTANTS['epsilon'])
            sensor_xy.append(config['y' + str(i)] * CONSTANTS['epsilon'])
            sensorPositions.append(sensor_xy)
            sensor_xy = []

        except:
            pass
        
    data = Data(sensorPositions, BOV.space, CONSTANTS['epsilon'])
    return calculate_confusion_matrix(data)
 
    
def get_confusion_matrix(config, 
                         run_on_colab = False,
                         epsilon = 1,
                         radius = 1,
                         print_epochs = True,
                         height = 8.0,
                         width = 8.0,
                         ROS = False,
                         multi_objective = False):
    
    global multi_objective_flag
    global CONSTANTS
    global runningOnGoogleColab
    runningOnGoogleColab = run_on_colab
    multi_objective_flag = multi_objective
    CONSTANTS = {
        'iterations': 1000,
        'initial_samples': 10,
        'epsilon': epsilon,
        'radius': radius,
        'height': height,
        'width': width,
        'max_sensors': 15
    }

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
                                        MaxSensors = 15
                                       )

    global BOV
    BOV =  BOVariables(
                       Data_path, 
                       CONSTANTS['epsilon'], 
                       15, 
                       15, 
                       CONSTANTS['radius'],
                       CONSTANTS['initial_samples'],
                       ROS = True
                      )
    
    return confusion_matrix(config)
'''    

def run(surrogate_type = 'prf',
        acq_optimizer_type = 'random_scipy',
        task_id = 'SPO',
        run_on_colab = False, 
        iteration = 1000, 
        epsilon = 1, # The distance between two nodes in the space grid:
        maxSensorNum = 15,  # max sensor numbers
        radius = 1, # radius of the motion sensors
        print_epochs = True,
        height = 8.0,
        width = 8.0,
        ROS = False,
        multi_objective = False,
        initial_state = 'fixed',
        input_sensor_types = {'model_motion_sensor': True, 'model_beacon_sensor': False},
      ):

    global multi_objective_flag
    global CONSTANTS
    global runningOnGoogleColab
    global sensor_types
    runningOnGoogleColab = run_on_colab
    multi_objective_flag = multi_objective
    CONSTANTS = {
        'iterations': iteration,
        'initial_samples': 10,
        'epsilon': epsilon,
        'radius': radius,
        'height': height,
        'width': width,
        'max_sensors': maxSensorNum
    }
    sensor_types = input_sensor_types
    sensorTypesNum = sum(1 for condition in sensor_types.values() if condition)

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
                                        MaxSensors = CONSTANTS['max_sensors']
                                       )

    global BOV
    BOV =  BOVariables(
                       Data_path, 
                       CONSTANTS['epsilon'], 
                       CONSTANTS['max_sensors'], 
                       CONSTANTS['max_sensors'], 
                       CONSTANTS['radius'],
                       CONSTANTS['initial_samples'],
                       ROS = True
                      )

    from openbox import sp
    

    # Define Search Space
    space = sp.Space()

    
    if (multi_objective_flag == False):
        list_of_variables = []
        for i in range(1, CONSTANTS['max_sensors'] + 1):
            if initial_state == 'fixed':
                x = sp.Int("x" + str(i), 1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon']), default_value=1)
                y = sp.Int("y" + str(i), 1, int((CONSTANTS['height'] - 1) / CONSTANTS['epsilon']), default_value=1)
                
                if sensorTypesNum > 1:
                    t = sp.Int("t" + str(i), 1, sensorTypesNum, default_value=random.randint(1, sensorTypesNum))
                
                else:
                    t = sp.Constant("t" + str(i), 1)
                
                list_of_variables.append(x)
                list_of_variables.append(y)
                list_of_variables.append(t)
                
            elif(initial_state == 'random'):
                x = sp.Int("x" + str(i), 1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon']), 
                           default_value=random.randint(1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon'])))
                
                y = sp.Int("y" + str(i), 1, int((CONSTANTS['height'] - 1) / CONSTANTS['epsilon']), 
                           default_value=random.randint(1, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon'])))
                
                if sensorTypesNum > 1:
                    t = sp.Int("t" + str(i), 1, sensorTypesNum, default_value=random.randint(1, sensorTypesNum))
                
                else:
                    t = sp.Constant("t" + str(i), 1)
                
                list_of_variables.append(x)
                list_of_variables.append(y)
                list_of_variables.append(t)
                
            else:
                raise NotImplementedError (initial_state + " is not implemented yet! Try using 'fixed' or 'random' values istead")


        space.add_variables(list_of_variables)
        history_list = []

        opt = Optimizer(
            function_to_be_optimized,
            space,
            max_runs = CONSTANTS['iterations'],
            surrogate_type = surrogate_type,
            acq_optimizer_type = acq_optimizer_type,
            time_limit_per_trial=31000,
            task_id = task_id,
        )
        history = opt.run()
            
    
    else:
        list_of_variables = []
        for i in range(1, CONSTANTS['max_sensors'] + 1):
            x = sp.Int("x" + str(i), 0, int((CONSTANTS['width'] - 1) / CONSTANTS['epsilon']), default_value=1)
            y = sp.Int("y" + str(i), 1, int((CONSTANTS['height'] - 1) / CONSTANTS['epsilon']), default_value=1)
            list_of_variables.append(x)
            list_of_variables.append(y)

        space.add_variables(list_of_variables)

        history_list = []

        opt = Optimizer(
            function_to_be_optimized,
            space,
            max_runs = CONSTANTS['iterations'],
            surrogate_type = surrogate_type,
            acq_optimizer_type = acq_optimizer_type,
            time_limit_per_trial=31000,
            task_id = task_id,
        )
        history = opt.run()
    
    return history