import socket
import csv
from nn import neural_net, load_This, LossHistory
from learningSARSA import train_net 
from SARSA_brain import SarsaTable
import numpy as np
import timeit
host = ''
port = 5560
storedValue = "Yo, what's up?"



def setupServer():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created.")
    try:
        s.bind((host, port))
    except socket.error as msg:
        print(msg)
    print("Socket bind complete.")
    return s

def setupConnection():
    s.listen(1) # Allows one connection at a time.
    conn, address = s.accept()
    print("Connected to: " + address[0] + ":" + str(address[1]))
    return conn

def GET():
    reply = storedValue
    return reply

def REPEAT(dataMessage):
    reply = dataMessage[1]
    return reply


def dataTransfer(conn):
    nn_param = [165, 150]
    params = {
    "batchSize": 100,
    "buffer": 5000,
    "nn": nn_param
    }
    activations = ['relu','relu','linear']
    model = neural_net(3, nn_param, activations)
    #Uncomment to load latest saved model
    model = load_This(model) 
    t = 0
    replay = []
    replaySARSA = []
    '''
    with open('resultsSARSA/replay-165-150-100-5000' + '.csv', newline = '') as replay_save:
        rd = csv.reader(replay_save) 
        counter = 1
        oldG = []
        actC = 0;
        awardz = 0;
        GenState = []
        for row in rd:
            if counter%4==1:
                for element in row:
                    oldG.append(float(element))
            if counter%4==2:
                actC = float(row[0])
            if counter%4==3:
                awardz = float(row[0])    
            if counter%4==0:
                for element in row:
                    GenState.append(float(element))
                replay.append((np.array(oldG),actC,awardz,np.array(GenState)))
                oldG = []
                GenState = []
            counter+=1
    for idx, set in enumerate(replay): #assuming your replay is not less than 2 elements, which is practical
        if idx>1:
            writeOldState = replay[idx][0]
            writeAction = replay[idx][1]
            writeRew = replay[idx][2]
            actionPrev = replay[idx-1][1]
            older_state = replay[idx-1][0]   
            replaySARSA.append((list(older_state), int(actionPrev), writeRew, list(writeOldState), int(action)))
    '''
    epsilon = 1
    fps = 0
#   Just stuff used below.
    max_car_distance = 0
    car_distance = 0

    data_collect = []
    '''
    with open('resultsSARSA/learn_data-165-150-100-5000' + '.csv', newline = '') as dataC_save:
        rdd = csv.reader(dataC_save)
        for row in rdd:
            data_collect.append([int(row[0]), int(row[1])])
    '''
    loss_log = []
    '''
    #double check the loss file
    with open('resultsSARSA/loss_data-165-150-100-5000' + '.csv', newline = '') as Loss_save:
        rddF = csv.reader(Loss_save)
        for row in rddF:
            loss_log.append([float(row[0])])
    print(loss_log)
    '''
    old_state = [0,0,0]
    RL = SarsaTable(actions=list((0,1,2)))
    start_time = timeit.default_timer()
    while True:
        # Receive the data
        data = conn.recv(1024) # receive the data
        data = data.decode('utf-8')
        # Split the data such that you separate the command
        # from the rest of the data.
        dataMessage = data.split(' ', 1)
        command = dataMessage[0]
        print(command)
        if command == 'GET':
            reply = GET()
        elif command == 'REPEAT':#will produce error if the command is REPEAT with nothing after
            reply = REPEAT(dataMessage)
        elif command == 'distance':#This command reads from the car sensors
            values = dataMessage[1].split() #dataMessage[1] is a string
            state = [0,0,0]#pad the state with zeros
            for idx,num in enumerate(values):
                state[idx]=float(num)#fill the state array with each of the values
            for idx, anM in enumerate(state):
                state[idx] = int(anM) - int(anM%10)
            print(format(state))
            #begin training
            if t == 0:#needs separate case for first test because the machine can't see into the future
                t+=1#increment time
                reply = str(np.random.randint(0,3))#choose random action
                old_state = state#create the first old_state to use in the next iteration
            else:
                t+=1  
                (reply, model, old_state, state, epsilon, replay, 
                loss_log, car_distance, data_collect, max_car_distance, fps, RL, replaySARSA) = train_net(model, params, old_state, state, t, epsilon, replay, 
                                                                                         loss_log, car_distance, data_collect, max_car_distance, fps, RL, replaySARSA)#do the thing
                print('Epsilon: '+str(epsilon))
                print('car_distance:' + str(car_distance))
                print('Max Distance:' + str(max_car_distance))
                if reply == 'KILL':
                    command = 'KILL'

        elif command == 'EXIT':
            print("Our client has left us")
            break
        elif command == 'KILL':
            print("Our server is shutting down.")
            s.close()
            break
        else:
            reply = 'Unknown Command'
        # Send the reply back to the client
        conn.sendall(str.encode(reply))
        #//print("Data has been sent!")
    conn.close()


        

s = setupServer()

while True:
    #try:
        conn = setupConnection()
        dataTransfer(conn)
    #except:
     #      print("Connection Failed")
      #     break
