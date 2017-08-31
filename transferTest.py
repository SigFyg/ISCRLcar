import csv
import numpy as np

replay = []
with open('results2/replay-' + '164-150-100-5000' + '.csv', newline = '') as replay_save:
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
print(replay)

replay.append(([1,1,1], 2, 15, [3,3,3]))

with open('results2/replay-' + '164-150-100-5000' + '.csv', 'w') as replay_save:
    wr = csv.writer(replay_save)
    for element in replay:
        wr.writerow(element[0])
        wr.writerow([element[1]])
        wr.writerow([element[2]])
        wr.writerow(element[3])

