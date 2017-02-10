# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:39:58 2017

@author: monica
"""
import cv2
import random

img1 = cv2.imread('pic.jpg',0)

ret,world = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
height, width = img1.shape

ar_sarsa1 = open('ar_sarsa1.txt','a')
ar_expsarsa1 = open('ar_expsara1.txt','a')
ar_sarsa2 = open('ar_sarsa2.txt','a')
ar_expsarsa2 = open('ar_expsarsa2.txt','a')
ar_q1 = open('ar_q1.txt','a')

#print height, width
#print world[width-1,height-1]

X_min = 0
Y_min = 0
X_max = width 
Y_max = height
epsilon = 0.05

#cv2.line(world, (X_min+2,Y_max-1), (X_min+25, Y_max-1), 1, 1)
#cv2.imshow('window',world)
#cv2.waitKey(500)

states = dict() #Key: poisition tuple Value: dictionary of right, left, Up, Down
                #Neighbours are list of tuple of position

reward = {'o':-2, 'u':-0.1, 'g':100}

actions = ['rt', 'lt', 'up', 'dw']

stateStatus = dict() # dictionary storing status of each position
                    #occupancy status where if neighbouring point is wall 
                    #occupancy status = 'o' else 'u'
                    #if the neighbour = goal, status = 'g'

stateActionValues = dict() # dictionary key: position,action, value: state value
                    #initialised to zero

goalStates = list()
startStates = list()

aveReturns_sarsa = list()
aveReturns_expsarsa = list()

maxSteps = 10000
pCorrectAction = 0.9

alpha_sarsa = 0.24
alpha_expsarsa = 0.27
alpha_q = 0.28

discount_factor = 0.997

def defineGoals_start():
    """
    populate the list goalStats with all position of goal points
    @param: void
    @return: void (Updates global parameter goalStates)
    """
    global goalStates, startStates
    
    for i in range(X_max-25,X_max-2):
        goalStates.append((i,Y_min))
        goalStates.append((i,Y_min+1))
        
    for j in range(X_min+2,X_min+25):
        startStates.append((j,Y_max-1))
        startStates.append((j,Y_max-2))

def neighboursDict(s):
    """
    Takes in state state and return neighbour dictionary as described above
    """
    neighbours = {'rt':[],'lt':[],'up':[],'dw':[]}
    
    x = s[0]
    y = s[1]
    
    if not ((x+1) > X_max-1 ):
        neighbours['rt'].append((x+1,y))
    if not ((x-1) < X_min ):
        neighbours['lt'].append((x-1,y))
    if not ((y+1) > Y_max-1 ):
        neighbours['dw'].append((x,y+1))
    if not ((y-1) < Y_min ):
        neighbours['up'].append((x,y-1))
    
    #print len(neighbours)
    
    return neighbours

def image2World():
    '''
    Takes in the image and build a maze world
    Black = 0, White = 255
    @param: void 
    @return: void (Updates the global parameter states, stateActionValues, stateStatus)
    '''
    global states, stateActionValues, stateStatus
    
    for i in range(X_max):
        for j in range(Y_max):
            
            states[(i,j)] = neighboursDict((i,j))
            
            for ele in actions:
                stateActionValues[((i,j),ele)] = 0
                
            if world[j,i] == 0:
                stateStatus[(i,j)] = 'o'
            elif world[j,i] == 255:
                stateStatus[(i,j)] = 'u'
                
            if (i,j) in goalStates:
                stateStatus[(i,j)] = 'g'

def winnerNeighbour(s,a):
    '''
    Takes in state: return Max Q valued neighbour with Q value
    @param: state --> tuple position
    @return: s', value --> tuple, float
    '''
    winningN = None
    winningAction = None
    values = list()
    
    for ele in actions:
        values.append(stateActionValues[(s,ele)])

    maxVal = max(values)
    
    winningN, winningAction = stateActionValues.keys()[stateActionValues.values().index(maxVal)]
    
    return winningN,winningAction
    
def eGreedyPolicy(s):
    '''
    Takes in current state and return action
    Policy is e-greedy so return random action with probability e
    and return greedy action with probability 1-e
    '''
    flag = 'N'
    a = 'N'
    
    if random.random() <= epsilon:
        flag = 'random'
    else:
        flag = 'greedy'
        
    if flag == 'greedy':
        n, win_a = winnerNeighbour(s,'n')
        a = win_a

    elif flag == 'random':
        index = random.randint(0,len(actions) - 1)
        a = actions[index]
    else:
        print 'Error \n'
        a = None
        
    return a

def eofEpisode(s, sCount):
    """
    takes in current state or step count and return if episode is over or not
    @param: current State s
    @return: Bool
    """
    if ((s in goalStates) or (sCount == maxSteps) or (stateStatus[s] == 'g')):
        return True
    else:
        return False
 
def takeAction(s,a):
    """
    Takes in state and action a as suggested by the policy
    and return next observed state and reward
    Stochasticity of the enviroment is modeled here.
    @param: state, action
    @return: next observed state, immediate reward
    """
    flag = 'N'
    
    if random.random() <= (1-pCorrectAction):
        flag = 'random'
    else:
        flag = 'correct'
        
    if flag == 'random':
        index = random.randint(0,len(actions) - 1)
        
        a_t = actions[index]
    elif flag == 'correct':
        a_t = a
    else:
        print 'Error in deciding the action \n'
    
    while(len(states[s][a_t]) == 0):
        flag = 'N'
        
        if random.random() <= (1-pCorrectAction):
            flag = 'random'
        else:
            flag = 'correct'
            
        if flag == 'random':
            index = random.randint(0,len(actions) - 1)
            
            a_t = actions[index]
            
        elif flag == 'correct':
            
            a_t = a
        else:
            print 'Error in deciding the action \n'  
        
        if len(states[s][a_t]) != 0:
            break
        
    s_next = states[s][a_t][0]
    nextStateStatus = stateStatus[s_next]
    r = reward[nextStateStatus]
    
    return s_next, r
        
def SARSA(numEpisodes):
    global stateActionValues, aveReturns_sarsa
    image2World() # Initialise Q(s,a)
    
    episodeCount = 0
    rewardPerEpidode = []
    
    while(episodeCount != numEpisodes):
        path_sarsa = list()
        reward_list = list()
        stepCount = 0
        
        start = startStates[random.randint(0,len(startStates)-1)]
        a = eGreedyPolicy(start)
        path_sarsa.append((start,a))
        
        curr_state = start
        curr_action = a
        
        while(not eofEpisode(curr_state,stepCount)):
            s_next, r = takeAction(curr_state,curr_action)
            reward_list.append(r)
            
            a_next = eGreedyPolicy(s_next)
            stateActionValues[(curr_state,curr_action)] = stateActionValues[(curr_state,curr_action)] + (alpha_sarsa * (r + discount_factor *stateActionValues[(s_next,a_next)] - stateActionValues[(curr_state,curr_action)] ))     
            path_sarsa.append((s_next,a_next))
            
            if r == -2:
                curr_state = curr_state
            else:
                curr_state = s_next
                
            curr_action = a_next
            stepCount += 1
            
        
        episodeCount += 1
        print 'Average reward of episode: ',episodeCount,'--->', sum(reward_list)/len(reward_list)
        #print "Sarsa Episode Count: ", episodeCount
        rewardPerEpidode.append(sum(reward_list)/len(reward_list))
        ar_sarsa1.writelines(str(sum(reward_list)/len(reward_list)))
    ar_sarsa1.close()
    return rewardPerEpidode

def EXP_SARSA(numEpisodes):
    global stateActionValues, aveReturns_expsarsa
    
    image2World() # Initialise Q(s,a)
    episodeCount = 0
    rewardPerEpisode = []
    
    while(episodeCount != numEpisodes):
        
        path_expsarsa = list()
        reward_list = list()
        stepCount = 0
        
        start = startStates[random.randint(0,len(startStates)-1)]
        curr_state = start
        
        while(not eofEpisode(curr_state,stepCount)):
            curr_action = eGreedyPolicy(curr_state) #IMP CHANGE HERE
            path_expsarsa.append((curr_state,curr_action))
            
            s_next, r = takeAction(curr_state,curr_action)
            reward_list.append(r)
            
            value_snext = 0
            
            for ele in actions:
                if s_next == states[curr_state][curr_action]:
                    value_snext += (pCorrectAction*stateActionValues[(s_next,ele)])
                else:
                    value_snext += ((1-pCorrectAction)*stateActionValues[(s_next,ele)])
            
            stateActionValues[(curr_state,curr_action)] = stateActionValues[(curr_state,curr_action)] + (alpha_expsarsa * (r + discount_factor *value_snext - stateActionValues[(curr_state,curr_action)] ))
            
            if r == -2:
                curr_state = curr_state
            else:
                curr_state = s_next
            stepCount += 1
            
        
        episodeCount += 1
        print 'Average reward of episode: ',episodeCount,'--->', sum(reward_list)/len(reward_list)
        #print "Exp-Sarsa Episode Count: ", episodeCount
        rewardPerEpisode.append(sum(reward_list)/len(reward_list))
        ar_expsarsa1.writelines(str(sum(reward_list)/len(reward_list)))
        
    ar_expsarsa1.close()
    return rewardPerEpisode


def Q_learning(numEpisodes):
    global stateActionValues, aveReturns_Q
    
    image2World() # Initialise Q(s,a)
    episodeCount = 0
    rewardPerEpisode = []
    
    while(episodeCount != numEpisodes):
        
        path_q = list()
        reward_list = list()
        stepCount = 0
        
        start = startStates[random.randint(0,len(startStates)-1)]
        curr_state = start
        
        while(not eofEpisode(curr_state,stepCount)):
            curr_action = eGreedyPolicy(curr_state) #IMP CHANGE HERE
            path_q.append((curr_state,curr_action))
            
            s_next, r = takeAction(curr_state,curr_action)
            reward_list.append(r)
            
            win_n, win_a = winnerNeighbour(curr_state,'n')
            max_Q = stateActionValues[(s_next,win_a)]
            
            stateActionValues[(curr_state,curr_action)] = stateActionValues[(curr_state,curr_action)] + (alpha_q * (r + discount_factor *max_Q - stateActionValues[(curr_state,curr_action)] ))
            
            if r == -2:
                curr_state = curr_state
            else:
                curr_state = s_next
            stepCount += 1
            
        
        episodeCount += 1
        print 'Average reward of episode: ',episodeCount,'--->', sum(reward_list)/len(reward_list)
        
        rewardPerEpisode.append(sum(reward_list)/len(reward_list))
        ar_q1.writelines(str(sum(reward_list)/len(reward_list)))
        
    ar_q1.close()
    return rewardPerEpisode






if __name__ == '__main__':
    defineGoals_start()

    rewards_epi_sarsa = SARSA(80)
    print rewards_epi_sarsa
    print '----------------------------------------------'

    rewards_epi_expsarsa = EXP_SARSA(80)
    print rewards_epi_expsarsa
    print '----------------------------------------------'

    rewards_epi_q = Q_learning(30)
    print rewards_epi_q
    
    