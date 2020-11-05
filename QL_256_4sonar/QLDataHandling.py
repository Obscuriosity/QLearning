'''
QLDataHandling for Q Learning programs
'''
import os.path
import numpy as np
from numpy import save
from numpy import load
import pickle
import matplotlib.pyplot as plt

NAME = 'QL_256_4sonar'
LOG = (f'{NAME}_log.txt')
STATES_LIST = (f'States_List_{NAME}.npy')
QTABLE = (f'Q_Table_{NAME}.npy')
DICTIONARY = (f'{NAME}_Stats.pkl')
QuTABLE = (f'Qu_Table_{NAME}.npy')

def LoadLog():
    t = int(0) # Variable for timestep count

    tlog = open(LOG) # Open File for reading to retrieve data
    tlog.readline()        #Read first line and do nothing with it
    oldt = tlog.readline() # Read file and assign iteration data to variable
    tlog.close() # Close file
    print(oldt)
    t = int(oldt) # convert numeric data into int
    return t

def LoadQTable():
    Q = load(QTABLE)
    return Q

def LoadStatesList():    
    states = load(STATES_LIST)
    return states

def LoadQuTable():
    QuT = load(QuTABLE) # UCB data
    return QuT

def LoadDictionary():
    Dictionary = pickle.load(open(DICTIONARY, "rb"))
    return Dictionary

def SaveData(t, Q, Qu, DictData):
    save(QTABLE, Q)
    save(QuTABLE, Qu)
    if t % 1000 == 0:
        save((f'QTables/{NAME}_{t}.npy'), Q)
    
    t = str(t) # convert back to string for writing
    tlog = open(LOG, "w+") # open file for writing
    tlog.write("Itteration Number :\n") # Lets write what the file is about
    tlog.write(t) # write new data to file              # insert variables after conversion to string
    tlog.close()  # Close file
    t = int(t)
    
    # Save Dictionary Data
    StatDict = open(DICTIONARY, "wb")
    pickle.dump(DictData, StatDict)
    StatDict.close
    
    print("Time Steps :", t)
    print("Data Saved.")

def SaveGraph(t, alpha, gamma):
    aggr_rewards = pickle.load(open(DICTIONARY, "rb"))
    title = (f'{NAME}, alpha, {alpha},  gamma, {gamma}')
    plt.figure(figsize=(22, 12))
    plt.plot(aggr_rewards['t'], aggr_rewards['eps'], 'r:', label="epsilon")
    plt.plot(aggr_rewards['t'], aggr_rewards['avg'], label="average rewards")
    plt.plot(aggr_rewards['t'], aggr_rewards['max'], label="max rewards")
    plt.plot(aggr_rewards['t'], aggr_rewards['min'], label="min rewards")
    plt.xlabel('Iterations')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend(loc=3)
    plt.savefig(f'QTables/Graph_{NAME}_{t}.png')
    print("Graph Saved")
    #plt.show()
    
def mapp(x, in_min, in_max, out_min, out_max):
    # Function to map reward values ie - r = round(mapp(a, 0, 75, 0, 10), 4)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def CreateData():
    sonar_range = 2
    startVal = 1.0

    Q_table = np.array([[startVal, startVal, startVal]])
    States_List = []
    for i in range(0, sonar_range):
        for j in range(0, sonar_range):
            for k in range(0, sonar_range):
                for l in range(0, sonar_range):
                    for m in range(0, sonar_range):
                        for n in range(0, sonar_range):
                            for o in range(0, sonar_range):
                                for p in range(0, sonar_range):
                                    newState = (i, j, k, l, m, n, o, p)
                                    States_List.append(newState)
    print('states is ', len(States_List), " long.")
    
    # Create Q Table
    for o in range(0, len(States_List)-1):
        newQ = np.array ([startVal, startVal, startVal])  # create new action value list
        Q_table = np.vstack((Q_table, newQ))
    print("Q Table is ", len(Q_table), " long.")
    
    # Create UCB Qu Table with 256states 3 actions and 
    # number of times picked (0) and running total of rewards (1)
    actions = 3
    Qu = np.zeros((len(States_List), actions, 2))
    
    log = open(LOG, "w")
    log.write("Itteration Number :\n") # Lets write what the file is about
    log.write("0\n")
    log.close()

    save(QTABLE, Q_table)
    save(STATES_LIST, States_List)
    save(QuTABLE, Qu)

    print("Q Table Created\nState List Created\nLog created")
    
    cwd = os.getcwd() # Get current working directory
    dir = os.path.join(cwd,'QTables') # Make directory path
    if not os.path.exists(dir) == True: # If path does not exist Create it
        os.mkdir(dir)
        print(dir, "- Made")
    t = 'EMPTY'
    save((f'QTables/{NAME}_{t}.npy'), Q_table) # Added zeros so as not to overwrite current file if one in use. Delete '_0000' to use
    
    # Create and save new dictinary
    Dictionary = {'t': [], 'avg': [], 'max': [], 'min': [], 'eps': []}
    StatDict = open(DICTIONARY, "wb")
    pickle.dump(Dictionary, StatDict)
    StatDict.close
