import SIM_SIS_Libraries.SensorsClass as sc

space = [(0.0, 0.0), (0.0, 8.0), (8.0, 8.0), (8.0, 0.0)]
rooms = {'bedroom': [(3.9, 0.0), (8.0, 4.4)],
         'livingroom': [(0.0, 1.9), (6.3, 6.7)],
         'diningroom': [(0.0, 3.0), (2.9, 8.0)],
         'kitchen': [(0.0, 3.0), (2.9, 8.0)],
         'bathroom': [(6.1, 3.2), (8.0, 6.7)],
		 'storage': [(2.8, 6.4), (8.0, 8.0)]} 



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