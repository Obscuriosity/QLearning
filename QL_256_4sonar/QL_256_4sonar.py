'''
QL_256_4sonar
    Robot Q Learning experiment with 256 states and 3 actions.
    States are 0 or 1 on four sonar to imply objects are within a threshold, I may make tis one less than 36cm.
    Actions are forward, spin left and spin right.
    States will be based on current state and previous state giving a greater perception of sequence.
    Added UCB (Upper Confidence Bound) policy, to see if I can notice any Improvement.
    The UCB data list Qu will contain the number of times an action is taken and also be used to store total reward to get average reward

    getReward - removed the -5 penalty for turning and reduced the -100 penalty for crashing to a token -3.
'''

# Import Stuff
import RPi.GPIO as GPIO
import time
import serial
import math
import random
import numpy as np
import QLDataHandling as DH
import os.path
import matplotlib.pyplot as plt
import pickle

# Serial Setup
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
arduino.flush()

t = int(0) # Variable for timestep count

if os.path.isfile(DH.LOG) == True: #'QL_16_Explore_log.txt'
    print ("Loading Lists...")
    t = DH.LoadLog()
else:
    print("Creating Lists...")
    DH.CreateData()
    t = 0

Q = DH.LoadQTable()
states = DH.LoadStatesList()
Qu = DH.LoadQuTable()
aggr_rewards = DH.LoadDictionary()

# For stats
STATS_EVERY = 100
rewards = []

# Set up motors and GPIO pins
GPIO.setmode(GPIO.BCM) # Use Broadcom pin numbering
GPIO.setwarnings(False)
GPIO.setup(22, GPIO.OUT) # Left Motor Forward
GPIO.setup(23, GPIO.OUT) # Left Motor Backward
GPIO.setup(27, GPIO.OUT) # Right Motor Forward
GPIO.setup(18, GPIO.OUT) # Right Motor Backward
GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Pause switch
leftFor = GPIO.PWM(22, 50)
leftBac = GPIO.PWM(23, 50)
rightFor = GPIO.PWM(27, 50)
rightBac = GPIO.PWM(18, 50)

# initiate Motors
leftDutyCycle, rightDutyCycle = 0, 0
leftFor.start(leftDutyCycle)
leftBac.start(0)
rightFor.start(rightDutyCycle)
rightBac.start(0)

button = 4 # Pause Switch GPIO 4
pause = 0  # Paused / Resume state

# Lets have a function to stop the motors
Stopped = False
def Stop():
    print("STOPPED")
    global leftDutyCycle, rightDutyCycle
    leftDutyCycle, rightDutyCycle = 0, 0
    leftFor.start(leftDutyCycle)
    leftBac.start(0)
    rightFor.start(rightDutyCycle)
    rightBac.start(0)

def Pause(): # Pause routine, Uses sleep
    global pause
    if GPIO.input(button) == 1:
        time.sleep(.3)
        if pause == 0:
            pause = 1
            Stop()
            DH.SaveData(t, Q, Qu, aggr_rewards)
            DH.SaveGraph(t, alpha, gamma)
            print("Paused")
        elif pause == 1:
            pause = 0
            startT = t + 1
            print("Resumed")

dataList = [] # this is where we store the data from the arduino
noData = True # Boolean to let us know if we've received any info from the Arduino

def Serial():  # Communicate with arduino to read encoders, bumpers and sonars
    #print("Serial")50
    try:
        global noData
        arduino.write(b"Send\n")
        data = arduino.readline().decode('utf-8').rstrip()
        if len(data) < 1:
            print("No data ", data)
            noData = True
        else:
            global dataList
            dataList = data.split(",") # split at comma and make a list
            dataList = list(map(int, dataList)) # Convert string to ints
            noData = False
            #print("DATA", dataList)
            # 0 = Left, 1 = Front and 2 = Right Bumper,
            # 3 = Left Sonar No1, 4 = Left Secondary Sonar No2, 5 = Front Left Secondary Sonar No 3, 6 = Front Left Sonar No 4
            # 7 = Front Right Sonar No5, 8 = Front Right Secondary Sonar No6, 9= Right Secondary Sonar No7, 10 = Right Sonar No 8.
            # 11 = left forward 12 = left back 13 = right forward 14 = right back encoders
    except:
        print('no Connection')

SHORT_DISTANCE = 15 # threshold in cm over which obstacles are ignored

def SONAR(position): # Retrieve the state of individual Sonar
    global dataList, LONG_DISTANCE
    '''
    # Four Sonar Script
    if position == 0:
        distance = dataList[3] # Get Left Sonar reading from list
    if position == 1:
        distance = dataList[4] # Get Left Front Sonar reading from list
    if position == 2:
        distance = dataList[5] # Get Right Sonar reading from list
    if position == 3:
        distance = dataList[6] # Get Right Front Sonar reading from list
    '''
    # Eight Sonar Script
    if position == 0:
        if dataList[4] > 0 and dataList[3] == 0: # if Left Secondary(4) registers an obstacle which Left(3) doesnt, Override.
            distance = dataList[4]
        elif dataList[3] > dataList [4] and dataList[4] > 0:
            distance = dataList[4]
        else:
            distance = dataList[3] # Else Get Left Sonar reading from list
    if position == 1:
        if dataList[5] > 0 and dataList[6] == 0: # if Front Left Secondary(5) registers an obstacle which Front Left(6) doesn't, override
            distance = dataList[5]
        elif dataList[6] > dataList [5] and dataList[5] > 0:
            distance = dataList[5]
        else:
            distance = dataList[6] # Else Get Front Left Sonar reading from list
    if position == 2:
        if dataList[8] > 0 and dataList[7] == 0: # if Front Right Secondary(8) registers an obstacle which Front Right(7) doesn't, override
            distance = dataList[8]
        elif dataList[7] > dataList [8] and dataList[8] > 0:
            distance = dataList[8]
        else:
            distance = dataList[7] # Get Front Right Sonar reading from list
    if position == 3:
        if dataList[9] > 0 and dataList[10] == 0: # if Right Secondary(9) registers an obstacle which Right(10) doesn't, Override
            distance = dataList[9]
        elif dataList[10] > dataList [9] and dataList[9] > 0:
            distance = dataList[9]
        else:
            distance = dataList[10] # else Get Right Sonar reading from list

    # Return a state based on distance to objects.
    if distance > SHORT_DISTANCE or distance < 1: # newPing returns distances over 100cm as 0
        State = 0
    if distance < SHORT_DISTANCE + 1 and distance > 0:
        State = 1
    return int(State)

lastState = [0,0,0,0]

def getState(): # Returns state of the percieved world as a list
    global states, lastState
    S1 = SONAR(0)  # read left sonar and get number from 1 to 4
    S2 = SONAR(1)  # read left front sonar
    S3 = SONAR(2)  # read right front sonar
    S4 = SONAR(3)   # read right sonar
    newState = [S1, S2, S3, S4]
    if t <= startT:
        lastState = newState
    currentState = np.hstack((newState, lastState))
    lastState = newState
    s = np.where((states == currentState).all(axis=1))#
    print ("New State = ", currentState)
    #print ("State, s = ", s)
    return s                   # return list index to retieve data about state and action values (Q values)

# Number of Actions = 3 # drive forward at 50%, turn Left or turn Right.
def getAction(s, epsilon): # pass the s index of Q table and epsilon, to get maxQ make epsilon 1
    global Q, Qu
    action = 0
    randVal = 0
    #Epsilon Greedy - if epsilon is below 1 there is a chance of a random allowed action being chosen (exploration)
    randVal = random.randrange(1,101)
    if randVal <= (1-epsilon)*100:
        action = np.argmax(Q[s])
    else:
        action = random.randrange(0,3)
        #print("Random Action = ", action, ", Random Value = ", randVal, ", Epsilon = ", epsilon)
    Qu[s, action, 0] += 1 # Add to running total number of times chosen

    return(action)

def Forward():
    leftFor.ChangeDutyCycle(leftDutyCycle)
    leftBac.ChangeDutyCycle(0)
    rightFor.ChangeDutyCycle(rightDutyCycle)
    rightBac.ChangeDutyCycle(0)

def Reverse():
    leftFor.ChangeDutyCycle(0)
    leftBac.ChangeDutyCycle(50)
    rightFor.ChangeDutyCycle(0)
    rightBac.ChangeDutyCycle(50)

def SpinLeft():
    leftFor.ChangeDutyCycle(leftDutyCycle)
    leftBac.ChangeDutyCycle(0)
    rightFor.ChangeDutyCycle(0)
    rightBac.ChangeDutyCycle(rightDutyCycle)

def SpinRight():
    leftFor.ChangeDutyCycle(0)
    leftBac.ChangeDutyCycle(leftDutyCycle)
    rightFor.ChangeDutyCycle(rightDutyCycle)
    rightBac.ChangeDutyCycle(0)

def Act(action):
    global leftDutyCycle, rightDutyCycle
    leftDutyCycle, rightDutyCycle = 50, 50
    if action == 0: # Turn Left
        SpinLeft()
    if action == 1: # Drive Forward
        Forward()
    if action == 2: # Turn Right
        SpinRight()
crashed = False

REWARD_LIST = np.array([-1, -2, -2, -1, -1, -2, -2, -1])

def getReward(newState):
    global crashed, states, s, a, REWARD_LIST, Qu
    r = 0
    reward = np.multiply(states[newState], REWARD_LIST)
    #print('State Reward = ', reward)
    r += np.sum(reward)
    if np.sum(reward) == 0:
        r += 1
    if crashed == True:
        r -= 3
    #print ("Reward = ", r)
    r = round(r, 2)
    Qu[s, a, 1] += r # Add to running total reward list

    return (r)

def QLearn():
    global Q, s, states, alpha, gamma
    newS = getState() # newS = index of state in states list
    r = getReward(newS) # get reward based on action and distance from obstacles
    #print ("Old State = ", states[s])
    max_future_Q = np.max(Q[newS]) # Get Q Value of optimum action.
    currentQ = Q[s,a]
    #print ("currentQ = ", currentQ)
    #newQ = (1 - alpha) * currentQ + alpha * (r + gamma * max_future_Q)  # got from https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
    #print ("currentQ = ", currentQ)
    newQ = currentQ + alpha * (r + gamma * max_future_Q - currentQ) # Bellman equation, the heart of the Reinforcement Learning program
    newQ = np.round(newQ, 2) # round down floats for a tidier Q Table
    Q[s, a] = newQ
    s = newS
    #print ("NewQ = ", newQ)
    LogDict(r)

def LogDict(r):
    global  epsilon
    # Log rewards to monitor progress
    rewards.append(r)
    if not t % STATS_EVERY:
        average_reward = sum(rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_rewards['t'].append(t)
        aggr_rewards['avg'].append(average_reward)
        aggr_rewards['max'].append(max(rewards[-STATS_EVERY:]))
        aggr_rewards['min'].append(min(rewards[-STATS_EVERY:]))
        aggr_rewards['eps'].append(epsilon)

epsilon = 0.1
EPSILON_END = 100000
EPSILON_DECAY = epsilon/EPSILON_END
if 0 <= t < EPSILON_END:
    epsilon -= EPSILON_DECAY * t
else:
    epsilon = 0

#Computational parameters
alpha = 0.1    #"Forgetfulness" weight or learning rate.  The closer this is to 1 the more weight is given to recent samples.
gamma = 0.9    #look-ahead weight or discount factor 0 considers new rewards only, 1 looks for long term rewards
# Step/time parameters
lasttime = time.time() # Variable to store time for timesteps
step = 0
previousStep = 0

startT = t + 1 # Set starting itteration number based on info saved in log

print("Start")
print("setting up Serial")
time.sleep(2)
print("Getting Data")
while noData == True:
    Serial()
print("DATA", dataList)
s = getState() # s = index of state in states list

while True:

    Pause()
    if pause == 1:
        pass
    else:
        if  time.time() > lasttime + 0.1: # .1 is closer to the actual iteration time of the pi
            lasttime = time.time()
            step = time.time() - previousStep
            #print("Iteration time = ", round(step, 4))
            previousStep = time.time()
            if t % 1000 == 0:
                DH.SaveData(t, Q, Qu, aggr_rewards)
            while noData == True:
                Stop()
                Serial()
            Serial()
            LB = dataList[0]
            FB = dataList[1]
            RB = dataList[2]
            if LB == 0 or FB == 0 or RB == 0: # if bumpers are hit, Stop.
                if Stopped == False:
                    Stop()
                    Stopped = True
                    crashed = True
                    DH.SaveData(t, Q, Qu, aggr_rewards) #  Lets take this time to save iterations and Q values.
                    QLearn()
                    #Reverse from obstruction
                    Reverse()
                    time.sleep(.1)
                    startT = t + 1
                    s = getState()
                    Stopped = False
                    crashed = False

            else:
                t += 1
                if epsilon > 0:
                    epsilon -= EPSILON_DECAY # reduces to 0 over 10,000 steps
                #print("Iteration = ", t)
                Q[0,1] = 0 # reset forward on state 0 to 0

                if t > startT: # on the first time through the loop there will be no reward or previous states or actions
                    QLearn()
                a = getAction(s, epsilon)   # getAction will find an action based on the index s in the Q list and exploration will be based on epsilon
                #print("Action = ", a)
                Act(a)
