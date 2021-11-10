import SIM_SIS_Libraries.SensorsClass as sc

space = [(0.0, 0.0), (0.0, 10.6), (6.6, 10.6), (6.6, 0.0)]
rooms = {'bedroom': [(0.0, 0.0), (3.0, 4.5)],
         'livingroom': [(3.1, 0.0), (7.5, 11.5)],
         'diningroom': [(3.1, 0.0), (7.5, 11.5)],
         'kitchen': [(0.0, 0.0), (7.5, 11.5)],
         'bathroom': [(0.0, 4.3), (2, 7.6)]} 



def GetUsersParameters():
    '''Gets and parses a file of user defined parameters. The output is (list of [type, number])'''
    #TODO: Real implementation!
    types = [['motion sensors', 14], ['beacon sensors', 14]]
    distribution = {'bedroom': [3, 3],
         'livingroom': [3, 3],
         'diningroom': [2, 2],
         'kitchen': [4, 4],
         'bathroom': [2, 2]
    }
    
    return types, distribution

def ParseWorld(simworldname):
    '''Gets a sensor configuration space in XML and parses it. The output is (space dimensions, rooms dimensions)'''
    #TODO: Space and Rooms needs to be read from a file
        
    return space, rooms