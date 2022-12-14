import json

def readCONST():
    json_file = open('ReinforcementLearning/CONSTANTS.json')
    CONST = json.load(json_file)
    return CONST