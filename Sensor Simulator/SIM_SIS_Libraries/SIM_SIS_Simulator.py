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
def Initialization(sensorsConfiguration):
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
def InitializeDataset(sensorTypes, FDN):
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
    
    sensor_bins = [0] * sensorTypes[0][1]
    estimote_bins = [0] * sensorTypes[1][1]
    simulated_sensor_readings = [sensor_bins] * len(df_)
    simulated_estimote_readings = [estimote_bins] * len(df_)

#DONE
def VectorizeSensorReadings(fs, time):
    sensor_bins = [0] * sensorsTypes[0][1]
    estimote_bins = [0] * sensorsTypes[1][1]
    
    if len(fs) == 0:
        simulated_sensor_readings[time] = list(map(sum, zip(simulated_sensor_readings[time], sensor_bins))) 
        simulated_estimote_readings[time] = list(map(sum, zip(simulated_estimote_readings[time], estimote_bins))) 
        
    else:
        for sensor in fs:
            if (sensor.sensor_type == 'motion_sensor'):
                # snum = int(sensor.sensor_id.replace('sensor',''))
                sensor_bins[int(sensor.sensor_id)] = 1

            elif (sensor.sensor_type == 'beacon_sensor'):
                estimote_bins[int(sensor.sensor_id)] = 1
                
        simulated_sensor_readings[time] = list(map(sum, zip(simulated_sensor_readings[time], sensor_bins))) 
        simulated_estimote_readings[time] = list(map(sum, zip(simulated_estimote_readings[time], estimote_bins))) 
            
#DONE
def RegionOfSimilarity(exactLocation, epsilon):

    new_x =  uniform(max(exactLocation[0] - epsilon, 0), exactLocation[0] + epsilon)
    new_y =  uniform(max(exactLocation[1] - epsilon, 0), exactLocation[1] + epsilon)
    
    
    room, name = FindAgentsRoom(exactLocation[0], exactLocation[1])
    
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
def FindAgentsRoom(x, y):
    for room in rooms:
        if (x >= rooms[room][0][0] and x <= rooms[room][1][0] and
            y >= rooms[room][0][1] and y <= rooms[room][1][1]):
            return rooms[room], room

    return None
    
def SimulateSensorReadings(simulateMotionSensors, simulateEstimotes, t, i, agent1Loc, agent2Loc = None):
    myfs = []
    for sensor in sensors_list:
        if (simulateEstimotes == False and sensor.sensor_type == "beacon_sensor"):
            continue
            
        if (simulateMotionSensors == False and sensor.sensor_type == "motion_sensor"):
            continue
    
        if (simulateMotionSensors):
            if (plotflag and sensor.sensor_type == "motion_sensor"):
                pp = ax.plot(float(sensor.x) / 100, float(sensor.y) / 100 , marker='.', color='k', lw=5)

        if (simulateEstimotes):
            if (plotflag and sensor.sensor_type == "beacon_sensor"):
                pp = ax.plot(float(sensor.x) / 100, float(sensor.y) / 100 , marker= '2' , color='k', lw=5)                

        circ = Circle((float(float(sensor.x) / 100), float(float(sensor.y) / 100)), float(float(sensor.sensing_area) / 100))

        
        if (circ.contains_point([agent1Loc[0], agent1Loc[1]]) and (RecContains(agent1Loc[0], agent1Loc[1], sensor.room) or sensor.sensor_type == "beacon_sensor")):
            

                
            no_event_flag = 0
            event = ec.Event()
            event.sensor = sensor.sensor_id #SensorId that created the event
            event.data = "TRUE"  #data
            event.hash = "|hash|" #hash
            event.source = "xmlFile" #where is coming from
            event.timestamp = t
            event.sensorType = sensor.sensor_type #type of sensor

            if (plotflag):
                p3 = ax.plot(float(float(sensor.x) / 100), float(float(sensor.y) / 100) , marker='>', color='r', lw=10)

            # current_location = run_localization(event)
            
            if (float(FiringProbability(sensor, agent1Loc)) > random.uniform(0, 1)):
                myfs.append(sensor)

            
    # if (simulateMotionSensors):
    #     real_fired = [ii for ii, e in enumerate(df_.gt_motion_readings[i]) if e == 1]
    #     if (plotflag):
    #         for rf in real_fired:
    #             sid = str(motion_sensors[rf])
    #             for s in sensors_list:
    #                 if (s.sensor_type == "MotionSensorBinary"):
    #                     if (s.sensor_id == sid):
    #                         p4 = ax.plot(float(float(s.x) / 100), float(float(s.y) / 100) , marker='<', color='b', lw=5)


    # if (simulateEstimotes):
    #     real_fired = [ii for ii, e in enumerate(df_.gt_estimote_readings[i]) if e == 1]
    #     if (plotflag):
    #         for rf in real_fired:
    #             sid = estimote_names[rf]
    #             for s in sensors_list:
    #                 if (s.sensor_type == "BeaconSensor"):
    #                     if (s.sensor_id == sid):
    #                         # print(sid, df_.sthx[i], df_.sthy[i])
    #                         # sensed_areas[s.sensor_id].append((df_.sthx[i], df_.sthy[i]))
    #                         p4 = ax.plot(float(float(s.x) / 100), float(float(s.y) / 100) , marker='<', color='b', lw=5)

    no_event_flag = 0

    VectorizeSensorReadings(myfs, i) 

    if (plotflag):
        xlim=(0, 6)
        ylim=(0, 10)
        p1 = ax.plot(agent1Loc[0], agent1Loc[1], marker='+', color='k', lw=10)

        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().invert_yaxis()
        plt.show()
        fig.canvas.draw()
        ax.cla()
        ax.imshow(img, extent=[0, 6.6, 0, 10])


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
        
        if (xtrace < 0):
            xtrace = abs(xtrace)
        
        if (ytrace < 0):
            ytrace = abs(ytrace)
            
        
        agent1Loc = [xtrace, ytrace]
        
        agent1Loc_previous = [0, 0]
        epsilon = 0.7
        
        if (not agent1Loc_previous == agent1Loc):
            agent1Loc_rnd = RegionOfSimilarity(agent1Loc, epsilon)
            agent1Loc_previous = agent1Loc
            
        timetoadd = df_.time[i]
    
        SimulateSensorReadings(simulateMotionSensors, simulateEstimotes, timetoadd, i, agent1Loc_rnd, None)

def CreateUltimateDataset(UDN, epoch):
    simulated_sensor_readings.append([0]*len(simulated_sensor_readings[0]))
    df_['motion sensors'] = [[float(j)/epoch for j in i] for i in simulated_sensor_readings[0: len(df_.x)]]
    
    simulated_estimote_readings.append([0]*len(simulated_estimote_readings[0]))
    df_['beacon sensors'] = [[float(j)/epoch for j in i] for i in simulated_estimote_readings[0: len(df_.x)]]
    
    df_.to_csv(UDN + ".csv", sep=',', index=False)

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

def RunSimulator(space, Rooms, agentTrace, sensorsConfiguration, simulateMotionSensorsflag, simulateEstimotesflag, plottingflag, iteration, Data_path):
    global fig
    global ax
    global img
    global df_
    global rooms
    rooms = Rooms
    Epoch = 1
    # FDN = Data_path + 'Data//Pandas Datasets//ARTEST'
    FDN = agentTrace
    UDN = Data_path + "/Results/DatasetForAgent" + str(iteration)
    
    sc = sensorsConfiguration[0]
    radius = sensorsConfiguration[1]
    
    simulateMotionSensors = simulateMotionSensorsflag
    simulateEstimotes = simulateEstimotesflag
    SwitchPlotFlag(plottingflag)
    
    Initialization(sc)
    InitializeDataset(sc[0], FDN)
    for epoch in range(Epoch):
        Initialization(sc)
        sensors_list = MakeSensorsList(sc[1], radius)
        if (plotflag):
            # %matplotlib notebook
            fig, ax = plt.subplots(figsize = (6.6, 10.5))
            img = plt.imread("Data//sc.png")        

        RunSimulation(FDN, simulateMotionSensors, simulateEstimotes)

    CreateUltimateDataset(UDN, Epoch)
    
    return df_
