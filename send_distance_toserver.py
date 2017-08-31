import socket
import RPi.GPIO as GPIO
import time
from rrb3 import *
rr = RRB3(9,6)

host='143.215.107.158'
port=5560

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((host,port))



def measureDistances():
    GPIO.setmode(GPIO.BCM)
    
    TRIG_MID = 18 
    ECHO_MID = 23

    TRIG_R = 20
    ECHO_R = 21

    TRIG_L = 19
    ECHO_L = 26

    GPIO.setup(TRIG_MID,GPIO.OUT)
    GPIO.setup(ECHO_MID,GPIO.IN)
    GPIO.setup(TRIG_R,GPIO.OUT)
    GPIO.setup(ECHO_R,GPIO.IN)
    GPIO.setup(TRIG_L,GPIO.OUT)
    GPIO.setup(ECHO_L,GPIO.IN)

    GPIO.output(TRIG_MID, False)
    GPIO.output(TRIG_R, False)
    GPIO.output(TRIG_L, False)
    print ("Waiting For Sensor To Settle")
    time.sleep(1)

    GPIO.output(TRIG_MID, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_MID, False)

    while GPIO.input(ECHO_MID)==0:
      pulse_start = time.time()
      
    while GPIO.input(ECHO_MID)==1:
      pulse_end = time.time()

    time.sleep(0.3)

    GPIO.output(TRIG_R, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_R, False)

    while GPIO.input(ECHO_R)==0:
      pulse_start_R = time.time()
      
    while GPIO.input(ECHO_R)==1:
      pulse_end_R = time.time()

    time.sleep(0.3)
    GPIO.output(TRIG_L, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_L, False)

    while GPIO.input(ECHO_L)==0:
      pulse_start_L = time.time()
      
    while GPIO.input(ECHO_L)==1:
      pulse_end_L = time.time()

    pulse_duration = pulse_end - pulse_start
    pulse_duration_R = pulse_end_R - pulse_start_R
    pulse_duration_L = pulse_end_L - pulse_start_L

    print(pulse_duration)

    distance = pulse_duration * 17150
    distance_R = pulse_duration_R * 17150
    distance_L = pulse_duration_L * 17150


    distance = round(distance, 2)
    distance_R = round(distance_R, 2)
    distance_L = round(distance_L, 2)


    print ("Distance from Middle Sensor:",distance,"cm")
    print ("Distance from Right Sensor:",distance_R,"cm")
    print ("Distance from Left Sensor:",distance_L,"cm")

    return distance, distance_R, distance_L

def measureGoal():
    return distanceToGoal, angleToGoal



while True:
    command=input("Enter your command: ")
    if (command == 'EXIT') or (command == 'KILL'):
        s.send(str.encode(command))
        break
    elif command=='distance':
        while True:
            print ("Distance Measurement In Progress")
            M,R,L = measureDistances()
            command = 'distance {0} {1} {2}'.format(M,R,L)
            s.send(str.encode(command))
            reply = s.recv(1024)
            print(reply.decode('utf-8'))
            reply = reply.decode('utf-8')
            if reply == '0':
                print('go left')
                rr.left(0.3)
            if reply == '1':
                print('go right')
                rr.right(0.3)
            if reply == '2':
                print('go forward')
                rr.reverse(0.5, .4) #this actually makes the car go forward. Don't ask
            if reply == '3':
                print('crash, go back')
                rr.forward(0.5, .4)
            if reply == 'KILL':
                rr.stop()
                break
            rr.stop()
        break
    
    elif command == 'toGoal':
        while True:
            print ("Distance Measurement In Progress")
            M,R,L = measureDistances()
            print ("Finding Goal")
            D,A = measureGoal()
            command = 'distance {0} {1} {2} {3} {4}'.format(M,R,L,D,A)
            s.send(str.encode(command))
            reply = s.recv(1024)
            print(reply.decode('utf-8'))
            reply = reply.decode('utf-8')
            if reply == '0':
                print('go left')
                rr.left(0.3)
            if reply == '1':
                print('go right')
                rr.right(0.3)
            if reply == '2':
                print('go forward')
                rr.reverse(0.5, .4) #this actually makes the car go forward. Don't ask
            if reply == '3':
                print('crash, go back')
                rr.forward(0.5, .4)
            if reply == 'KILL':
                rr.stop()
                break
            rr.stop()
        break
    rr.cleanup()
    s.send(str.encode(command))
    reply = s.recv(1024)
    print(reply.decode('utf-8'))

s.close()


