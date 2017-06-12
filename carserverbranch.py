import socket
from nn import neural_net, LossHistory
from learningForCar import train_net
import numpy as np
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
    nn_param = [164, 150]
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }
    model = neural_net(3, nn_param)
    t = 0
    replay = []
    epsilon = 1

    # Just stuff used below.
    max_car_distance = 0
    car_distance = 0
    #t = 0
    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').

    loss_log = []
    # A big loop that sends/receives data until told not to.
    old_state = [0,0,0]
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
            print(format(state))
            #begin training
            if t == 0:
                t+=1
                reply = np.random.randint(0,3)
                old_state = state
            else:
                reply = train_net(model, params, old_state, state, t, epsilon, replay, 
                          loss_log, car_distance, data_collect, max_car_distance)
            if reply == 'KILL'
                command == 'KILL'
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
