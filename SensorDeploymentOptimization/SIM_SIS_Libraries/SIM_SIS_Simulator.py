from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import xml.etree.ElementTree
import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time as pythonTimer
import scipy.stats as stats
from scipy.stats import multivariate_normal
import csv
import datetime
import datetime
import pandas as pd
import ast
import time
import math

# Evaluation tools:
# from dtw import dtw
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from operator import itemgetter
from random import randrange, uniform, randint
import random


import SIM_SIS_Libraries.ParseFunctions as pf
import SIM_SIS_Libraries.SensorsClass as sc
import SIM_SIS_Libraries.EventClass as ec

#DONE
def SwitchPlotFlag(b):
    global plotflag
    plotflag = b

#DONE    
def AgentNum(a):
    global agentsnum
    agentsnum = a

#DONE
def Initialization(sensorsConfiguration, simulateMotionSensors, simulateEstimotes):
    global groundtruth
    global motion_readings
    global sensorsTypes
    global grid_map
    global datasetname
    global distances
    global oracle
    global analyzer
    global dataset_times
    global dataset_readings
    global dataset
    global agentsnum
    global sensors_list
    global sr_matrix
    global indices_to_keep
    global gtx
    global gty
    global rsr
    global rooms
    global df_
    global gtt
    global human_readable_sensor_array
    global plotflag
    
    sensorsTypes = sensorsConfiguration[0]
    agentsnum = 1
    human_readable_sensor_array = []
    gtt = []
    gtx = []
    gty = []
    rsr = []
    
    distances = []

    oracle = {}
    oracle['time'] = []
    oracle['location'] = []

    analyzer = {}
    analyzer['time'] = []
    analyzer['gridmap'] = []
    dataset_times = []
    dataset_readings = []
    dataset = ['time', 'gt(x)', 'gt(y)']
    
    sensors_list = []
    #TODO: This is hard coded
    sr_matrix = np.zeros([1, 14])

#DONE
def InitializeDataset(sensorTypes, FDN, simulateMotionSensors, simulateEstimotes):
    global simulated_sensor_readings
    global simulated_estimote_readings
    global df_
    datasetColumns = ['time', 'x', 'y', 'activity']
    for st in sensorTypes:
        datasetColumns.append(st[0])
    
    tempdf = (pd.read_csv(FDN, index_col = False))
   
    
    df_ = pd.DataFrame(columns = datasetColumns)
    
    df_.time = tempdf.Time
    df_.x = tempdf.x
    df_.y = tempdf.y
    df_.activity = tempdf.Action
    
    if (simulateMotionSensors):
        try:
            motionNumbers = sensorsTypes[[a[0] for a in sensorsTypes].index('motion sensors')][1]
        except:
            motionNumbers = 0

        sensor_bins = [0] * motionNumbers        
        simulated_sensor_readings = [sensor_bins] * len(df_)
    
    if (simulateEstimotes):
        try:
            beaconNumbers = sensorsTypes[[a[0] for a in sensorsTypes].index('beacon sensors')][1]
        except:
            beaconNumbers = 0
            
        estimote_bins = [0] * beaconNumbers
        simulated_estimote_readings = [estimote_bins] * len(df_)
    
#DONE
def VectorizeSensorReadings(fs, time, agent1Loc, simulateMotionSensors, simulateEstimotes):
    if (simulateMotionSensors):
        try:
            motionNumbers = sensorsTypes[[a[0] for a in sensorsTypes].index('motion sensors')][1]
        except:
            motionNumbers = 0
            
        sensor_bins = [0] * motionNumbers
    
    if (simulateEstimotes):
        try:
            beaconNumbers = sensorsTypes[[a[0] for a in sensorsTypes].index('beacon sensors')][1]
        except:
            beaconNumbers = 0
            
        estimote_bins = [0] * beaconNumbers
        
    if len(fs) == 0:
        if (simulateMotionSensors):
            if (len(sensor_bins) > 0):
                simulated_sensor_readings[time] = list(map(sum, zip(simulated_sensor_readings[time], sensor_bins))) 
            else:
                simulated_sensor_readings[time] = []
        
        if (simulateEstimotes):
            if (len(estimote_bins) > 0):
                simulated_estimote_readings[time] = list(map(sum, zip(simulated_estimote_readings[time], estimote_bins))) 
            else:
                simulated_estimote_readings[time] = []
        
    else:
        for sensor in fs:
            if (sensor.sensor_type == 'motion_sensor'):
                if (simulateMotionSensors and motionNumbers != 0):
                    sensor_bins[int(sensor.sensor_id)] = 1

            elif (sensor.sensor_type == 'beacon_sensor'):
                if (simulateEstimotes and beaconNumbers != 0):
                    from math import dist
                    estimote_bins[int(sensor.sensor_id)] = sensor.MetersToRSSI(dist((sensor.x, sensor.y), (agent1Loc[0], agent1Loc[1])))
                    
        if (simulateMotionSensors):
            if (len(sensor_bins) > 0):
                simulated_sensor_readings[time] = list(map(sum, zip(simulated_sensor_readings[time], sensor_bins))) 
            else:
                simulated_sensor_readings[time] = []
        
        if (simulateEstimotes):
            if (len(estimote_bins) > 0):
                simulated_estimote_readings[time] = list(map(sum, zip(simulated_estimote_readings[time], estimote_bins)))
            else:
                simulated_estimote_readings[time] = []
            
def AddRandomnessToDatasets(epsilon, data_path, rooms):
    import pandas as pd
    import os
    
    directory = os.fsencode(data_path + 'Agent Trace Files/')
    
    for file in os.listdir(directory):
        # print(os.fsdecode(directory + file))
        data = pd.read_csv(os.fsdecode(directory + file), index_col = False)
        
        # print(data)
        
        for i in range(1, len(data)):
            xtrace = float(data.x[i])
            ytrace = float(data.y[i])

            agent1Loc = [xtrace, ytrace]
            agent1Loc_rnd = RegionOfSimilarity(agent1Loc, epsilon, rooms)
            data.at[i, 'x'] = agent1Loc_rnd[0]
            data.at[i, 'y'] = agent1Loc_rnd[1]

        data.to_csv(os.fsdecode(directory + file) + "UPDATED.csv", sep=',', index=False)
            
#DONE
def RegionOfSimilarity(exactLocation, epsilon, rooms):    
    new_x =  uniform(max(exactLocation[0] - epsilon, 0), exactLocation[0] + epsilon)
    new_y =  uniform(max(exactLocation[1] - epsilon, 0), exactLocation[1] + epsilon)
    
    
    room, name = FindAgentsRoom(exactLocation[0], exactLocation[1], rooms)
    
    new_x = min(max(room[0][0], new_x) + 0.1, room[1][0]) - 0.1
    new_y = min(max(room[0][1], new_y) + 0.1, room[1][1]) - 0.1
    
    return [new_x, new_y]

#DONE
def RecContains(x, y, sensor_room):
    for room in rooms:
        if (x >= rooms[room][0][0] and x <= rooms[room][1][0] and
            y >= rooms[room][0][1] and y <= rooms[room][1][1] and
        sensor_room == room):
            return True
    
    return False

#DONE
def FindAgentsRoom(x, y, rooms):
    for room in rooms:
        if (x >= rooms[room][0][0] and x <= rooms[room][1][0] and
            y >= rooms[room][0][1] and y <= rooms[room][1][1]):
            return rooms[room], room
        
    return None
    
def SimulateSensorReadings(simulateMotionSensors, simulateEstimotes, t, i, agent1Loc, action, agent2Loc = None):
    myfs = []
    for sensor in sensors_list:
        if (simulateEstimotes == False and sensor.sensor_type == "beacon_sensor"): continue
        if (simulateMotionSensors == False and sensor.sensor_type == "motion_sensor"): continue
    
        if (simulateMotionSensors and sensor.sensor_type == "motion_sensor"):
            if (plotflag):
                pp = ax.plot(float(sensor.x) / 100, float(sensor.y) / 100 , marker='.', color='k', lw=5)
                
            circ = Circle((float(float(sensor.x) / 100), float(float(sensor.y) / 100)), float(float(sensor.sensing_area) / 100))
                
            if (circ.contains_point([agent1Loc[0], agent1Loc[1]]) and 
                (RecContains(agent1Loc[0], agent1Loc[1], sensor.room))):
            
                no_event_flag = 0
                event = ec.Event()
                event.sensor = sensor.sensor_id #SensorId that created the event
                event.data = "TRUE"  #data
                event.hash = "|hash|" #hash
                event.source = "xmlFile" #where is coming from
                event.timestamp = t
                event.sensorType = sensor.sensor_type #type of sensor

                if (float(FiringProbability(sensor, agent1Loc)) > random.uniform(0, 1)):
                    myfs.append(sensor)

                if (0.1 > random.uniform(0, 1)):
                    myfs.append(sensor)

        if (simulateEstimotes and sensor.sensor_type == "beacon_sensor"):
            if (plotflag):
                pp = ax.plot(float(sensor.x) / 100, float(sensor.y) / 100 , marker= '2' , color='k', lw=5)                

                
            from math import dist
            no_event_flag = 0
            event = ec.Event()
            event.sensor = sensor.sensor_id
            event.data = sensor.MetersToRSSI(dist((sensor.x, sensor.y), (agent1Loc[0], agent1Loc[1])))
            event.hash = "|hash|" #hash
            event.source = "xmlFile" #where is coming from
            event.timestamp = t
            event.sensorType = sensor.sensor_type #type of sensor

            myfs.append(sensor)
        
    no_event_flag = 0

    VectorizeSensorReadings(myfs, i, agent1Loc, simulateMotionSensors, simulateEstimotes) 

    if (plotflag):
        # fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=80)
        from IPython import display
        import pylab as pl
        xlim=(0, 8.5)
        ylim=(0, 8.5)

        
        p1 = ax.plot(agent1Loc[0], agent1Loc[1], marker='+', color='k', lw=10)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().invert_yaxis()
        # plt.show()
        fig.canvas.draw()
        plt.title(str((agent1Loc[0], agent1Loc[1])) + " : " + action)
        # ax.cla()
        display.clear_output(wait=True)
        display.display(pl.gcf())
        

def FiringProbability(sensor, agentLocation):
    import math
    from scipy.stats import multivariate_normal
    
    sensor_location = [float(sensor.x)/100, float(sensor.y)/100]
    cov = [[float(float(sensor.sensing_area) / 100), 0],[0, float(float(sensor.sensing_area) / 100)]]        
    
    prob = multivariate_normal(agentLocation, cov)
    probCdf = prob.cdf(sensor_location)
    
    prob = multivariate_normal(sensor_location, cov)
    maxProb = prob.cdf(sensor_location)
    
    prob = multivariate_normal([float(sensor.x)/100 + float(float(sensor.sensing_area) / 100), float(sensor.y)/100 + float(float(sensor.sensing_area) / 100)], cov)
    minProb = prob.cdf(sensor_location)
    
    return float((probCdf - minProb) / (maxProb - minProb))
    
def RunSimulation(FDN, simulateMotionSensors, simulateEstimotes):     
    global df_
    for i in range(1, len(df_)):      
        no_event_flag = 1
        xtrace = float(df_.x[i])
        ytrace = float(df_.y[i])
        action = df_.activity[i]
         
        agent1Loc = [xtrace, ytrace] 
        agent1Loc_previous = [0, 0]
            
        timetoadd = df_.time[i]
        
        SimulateSensorReadings(simulateMotionSensors, simulateEstimotes, timetoadd, i, agent1Loc, action,  None)
    
def CreateUltimateDataset(UDN, epoch):
    simulated_sensor_readings.append([0]*len(simulated_sensor_readings[0]))
    df_['motion sensors'] = [[float(j)/epoch for j in i] for i in simulated_sensor_readings[0: len(df_.x)]]
    
    try:
        simulated_estimote_readings.append([0]*len(simulated_estimote_readings[0]))
        df_['beacon sensors'] = [[float(j)/epoch for j in i] for i in simulated_estimote_readings[0: len(df_.x)]]
        
    except:
        pass
                
    # print(df_)
      
    # TODO:    
    # df_.to_csv(UDN + ".csv", sep=',', index=False)

def MakeSensorsList(sensors, radius):
    radius = 1
    motionCounter = 0
    beaconCounter = 0
    for sensor_list in sensors:
        for sensor in sensor_list:
            if sensor[2] == 'motion sensors':
                this_sensor = sc.MotionSensorBinary(sensor[0][0], sensor[0][1], radius, sensor[1], motionCounter)
                motionCounter = motionCounter + 1
                sensors_list.append(this_sensor)
                

            elif sensor[2] == 'beacon sensors':
                this_sensor = sc.BeaconSensor(sensor[0][0], sensor[0][1], radius, sensor[1], beaconCounter)
                beaconCounter = beaconCounter + 1
                sensors_list.append(this_sensor)
                
    return sensors_list

def RunSimulator(space, Rooms, agentTrace, sensorsConfiguration, simulateMotionSensorsflag, simulateEstimotesflag, plottingflag, Data_path):
    global fig
    global ax
    global img
    global df_
    global rooms
    rooms = Rooms
    Epoch = 1
    # FDN = Data_path + 'Data//Pandas Datasets//ARTEST'
    FDN = agentTrace
    
    UDN = Data_path + "/Results/DatasetForAgent"
    
    sc = sensorsConfiguration[0]
    radius = sensorsConfiguration[1]
    
    simulateMotionSensors = simulateMotionSensorsflag
    simulateEstimotes = simulateEstimotesflag
    SwitchPlotFlag(plottingflag)
    
    Initialization(sc, simulateMotionSensors, simulateEstimotes)
    InitializeDataset(sc[0], FDN, simulateMotionSensors, simulateEstimotes)
    
    for epoch in range(Epoch):    
        Initialization(sc, simulateMotionSensors, simulateEstimotes)
        sensors_list = MakeSensorsList(sc[1], radius)
        
        if (plotflag):
            fig, ax = plt.subplots(figsize = (8.5, 8.5))

        RunSimulation(FDN, simulateMotionSensors, simulateEstimotes)
        
    CreateUltimateDataset(UDN, Epoch)
    
    
    return df_
