from ipywidgets import IntProgress
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
import CASAS.al as al
import pickle

class Chromosome:
    def __init__(self, *args, sensorTypesNum):
        self.sensorTypesNum = sensorTypesNum
        self.radius = 1
        if len(args) < 5:
            self.epsilon = args[3]
            self.initSensorNum = args[2]
            self.mode = args[0]
            self.space = args[1]
            self.placeHolders = []
            self.fitness = -1
            self.SensorPlaceHolderSetup()
            self.SensorConfigurationSetup()
              
        elif len(args) == 5:
            self.epsilon = args[3]
            self.placeHolders = []
            self.fitness = -1
            self.grid = args[0]
            self.mode = args[1]
            self.space = args[2]
            self.SensorPlaceHolderSetup()
            
        self.sensitivity = {'3': 'pressure',
                            '4': 'accelerometer',
                            '5': 'electricity'}
        
    def frange(self, start, stop, step):
        steps = []
        while start <= stop:
            steps.append(start)
            start +=step
            
        return steps
    
    def GoToNeighbour(self):
        nonzeroind = list(np.nonzero(self.grid)[0])
        zeroind = list(np.where(np.array(self.grid) == 0)[0])
        
        sensor = random.choice(nonzeroind)
        emptyPlace = random.choice(zeroind)
        
        # TODO: for mutiple sensor types:
        self.grid[sensor] = 0
        self.grid[emptyPlace] = 1
        

    def SensorPlaceHolderSetup(self):   
        Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
        
        import matplotlib.pyplot as plt
        
        for x in Xs:
          for y in Ys:
            self.placeHolders.append([x, y])

    def SensorConfigurationSetup(self):
        
        Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
        self.grid = np.zeros(len(Xs) * len(Ys)).tolist()
        
        i = 0
        while i < self.initSensorNum:
            cell = random.randrange(len(self.grid))
            if self.grid[cell] == 0:
                self.grid[cell] = 1
                i += 1
                
    def makeBoundaries(self, minValue, maxValue):
        m = self.epsilon * np.ceil(float(1/self.epsilon) * minValue)
        M = self.epsilon * np.floor(float(1/self.epsilon) * maxValue)
        return m, M
            
    def GetSensorConfiguration(self):
        from collections import Counter
        sensorLocations, sensorTypes = self.GetSensorLocations()
        _, rooms = pf.ParseWorld(simworldname = '')
        
        summaryDict = Counter(sensorTypes)

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

            if (sensorTypes[index] == 1):
                configurationDetails.append(tuple([loc, room, 'motion sensors']))

            elif (sensorTypes[index] == 2):
                configurationDetails.append(tuple([loc, room, 'beacon sensors']))
                
            elif (sensorTypes[index] == 3):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['3']]))
                
            elif (sensorTypes[index] == 4):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['4']]))
                
            elif (sensorTypes[index] == 5):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['5']]))

            else:
                configurationDetails.append(tuple([loc, room, 'motion sensors']))
        
        sensor_config = [[configurationSummary, [tuple(configurationDetails)]], self.radius]
        
        # print(sensor_config)
        
        return sensor_config


    def GetSensorLocations(self):
        sensorLocations = []
        sensorTypes = []
        for index, sensorIndicator in enumerate(self.grid):
            if (sensorIndicator > 0):
                sensorLocations.append(self.placeHolders[index])
                sensorTypes.append(sensorIndicator)

        return sensorLocations, sensorTypes


class SA:
    chromosomes = []
    def __init__(self, population, initializationMethod, Data_path, epsilon, initSensorNum, maxSensorNum, radius, ROS, learning_rate, sensorTypesNum):
        self.population = population
        self.mode = initializationMethod
        self.Data_path = Data_path
        self.epsilon = epsilon
        self.initSensorNum = initSensorNum
        self.maxSensorNum = maxSensorNum
        self.radius = radius
        self.learning_rate = learning_rate
        self.sensorTypesNum = sensorTypesNum
        
        self.sensor_distribution, self.types, self.space, self.rooms, self.agentTraces = self.ModelsInitializations(ROS)
        
        for i in range(population):
            self.chromosomes.append(Chromosome(self.mode, self.space, self.initSensorNum, self.epsilon, sensorTypesNum = self.sensorTypesNum))
          
    def GetNextGeneration(self, epoch):
        import copy 
        
        self.newConfigs = []
        self.newConfigs = copy.deepcopy(self.chromosomes)
        # self.newConfigs.append(copy.copy(self.chromosomes[0]))
        
        for config in self.newConfigs:
            config.GoToNeighbour()
        
        self.RunFitnessFunction(self.chromosomes, True, False, False, False, 1)
        self.RunFitnessFunction(self.newConfigs, True, False, False, False, 1)
        
        if self.newConfigs[0].fitness > self.chromosomes[0].fitness:
            self.chromosomes = copy.deepcopy(self.newConfigs)
            
        else:
            prob = random.uniform(0, 1)
            if prob <= self.learning_rate:
                self.chromosomes = copy.deepcopy(self.newConfigs)
                
        self.learning_rate = self.learning_rate / (epoch + 1)

    def ModelsInitializations(self, ROS):
        #----- Space and agent models -----: 
        simworldname = self.Data_path + '/Configuration Files/simulationWorld2.xml'
        agentTraces = []
        import os
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
        A = list(xs)
        A.sort()
        space = [A[-1], A[-2]]

        # print(space)

        # User parameters 
        types, sensor_distribution = pf.GetUsersParameters()

        roomsList = []
        for room in sensor_distribution:
            roomsList.append(room)
              
        return sensor_distribution, types, space, rooms, agentTraces

    def RunFitnessFunction(self, chroms, simulateMotionSensors, simulateEstimotes, simulateIS, Plotting, iteration):       
        for index, chromosome in enumerate(chroms):
            files = []

            all_sensors = set([])

            for agentTrace in self.agentTraces:
                filecoding = ' '#"_" + str(iteration) + "_c" + str(index + 1) + '(' + self.mode + ')'
                df_ = sim_sis.RunSimulator(self.space, 
                                           self.rooms, 
                                           agentTrace, 
                                           chromosome.GetSensorConfiguration(), 
                                           simulateMotionSensors, 
                                           simulateEstimotes, 
                                           simulateIS, 
                                           Plotting , 
                                           self.Data_path)

                dataFile, sensors = self.PreProcessor(df_)
                all_sensors.update(sensors)
                #self.D = dataFile
                files.append(dataFile)    

            all_sensors = list(all_sensors)

            chromosome.fitness = (al.leave_one_out(files, all_sensors)[0]) * 100
            if chromosome.fitness < 0:
                chromosome.fitness = 0

    def frange(self, start, stop, step):
        steps = []
        while start <= stop:
            steps.append(start)
            start +=step
            
        return steps

    def MakeSensorCombinations(self, start, end, epsilon, sensorType, room):
        a1, b1 = self.makeBoundaries(epsilon, start[0], end[0])
        a2, b2 = self.makeBoundaries(epsilon, start[1], end[1])    
        Xs = self.frange(a1, b1, epsilon)
        Ys = self.frange(a2, b2, epsilon)
        
        points = list(itertools.product(list(itertools.product(Xs, Ys)), [room], [sensorType[0]])) 
        C = itertools.combinations(points, distribution[room][types.index(sensorType)])

        return C
    
    
    def PreProcessor(self, df):
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
          
          for i in range(len(M)):
                sensorNames.append(self.Name(i, 'M'))
                if M[i] == 1:
                      if (previous_M != None):
                        if (previous_M[i] == 0):
                          MotionSensor_Names.append(self.Name(i,'M'))
                          MotionSensor_Message.append('ON')

                      else:
                        MotionSensor_Names.append(self.Name(i,'M'))
                        MotionSensor_Message.append('ON')

                if previous_M != None:
                      if M[i] == 0 and previous_M[i] == 1:
                        MotionSensor_Names.append(self.Name(i,'M'))
                        MotionSensor_Message.append('OFF')

          previous_M = M



          try:
              for i in range(len(I)):
                sensorNames.append(self.Name(i, 'IS'))
                if I[i] == 1:
                      if (previous_I != None):
                        if (previous_I[i] == 0):
                          ISSensor_Names.append(self.Name(i,'IS'))
                          ISSensor_Message.append('ON')

                      else:
                        ISSensor_Names.append(self.Name(i,'IS'))
                        ISSensor_Message.append('ON')

                if previous_I != None:
                      if I[i] == 0 and previous_I[i] == 1:
                        ISSensor_Names.append(self.Name(i,'IS'))
                        ISSensor_Message.append('OFF')

              previous_I = I

          except:
              pass 

          for m in range(len(MotionSensor_Names)):
            output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)

          for i_s in range(len(ISSensor_Names)):
            output_file.append(time +' '+ ISSensor_Names[i_s] + ' ' + ISSensor_Names[i_s] + ' ' + ISSensor_Message[i_s] + ' ' + Activity)

          for s in sensorNames:
              sensors.add(s)
                
        return output_file, list(sensors)
    

    def frange(self, start, stop, step):
        steps = []
        while start <= stop:
            steps.append(start)
            start +=step

        return steps
    
    def BestAnswer(self):
        bestAnswerIndex = 0
        for index, c in enumerate(self.chromosomes):
            if c.fitness > self.chromosomes[bestAnswerIndex].fitness:
                bestAnswerIndex = index

        return self.chromosomes[bestAnswerIndex]

    def AverageAnswer(self):
        return np.sum([c.fitness for c in self.chromosomes]) / len(self.chromosomes)

    def Name(self, number, typeSensor):
        if number < 10:
          return typeSensor + str(0) + str(number)
        else:
          return typeSensor + str(number)

    #converts epoch time to human readable
    def convertTime(self, posix_timestamp):
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
    
def run(run_on_colab = False, 
        iteration = 1000, 
        population = 1,
        epsilon = 1, # The distance between two nodes in the space grid:
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
        learning_rate = 1
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
        'max_IS_sensors': ISmaxSensorNum
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
    
    results = []
    best_configuration_history = []
    
    print('----- Running SimulatedAnnealing for epsilon = ' + str(epsilon))
    # f = IntProgress(min=0, max=iteration) # instantiate the bar
    # display(f) # display the bar
    
    sa = SA(1, 
            'expert', 
            Data_path, 
            epsilon, 
            CONSTANTS['max_LS_sensors'], 
            CONSTANTS['max_LS_sensors'], 
            radius, 
            ROS,
            learning_rate,
            LSsensorTypesNum + ISsensorTypesNum)
    
    # sa.RunFitnessFunction(True, False, False, 1)

    for epoch in range(iteration):
            print(epoch)
            # f.value += 1
            print('getting the next generation')
            sa.GetNextGeneration(epoch)
            # sa.RunFitnessFunction(True, False, False, 1)
            # sa.Selection()
            
            if (print_epochs == True):
                print("(epoch %d) ----- The best answer: {%f} with (%d) number of sensors" 
                      %(epoch + 1, 
                        sa.chromosomes[0].fitness,  
                        np.sum(sa.chromosomes[0].grid)))

            results.append([(c.fitness + (sum(c.grid) / 100) * 100, sum(c.grid)) for c in sa.chromosomes])
            best_configuration_history.append(sa.chromosomes[0])
            
    return results, best_configuration_history