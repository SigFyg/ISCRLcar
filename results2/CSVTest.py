import csv
with open('learn_data-164-150-100-5000.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))
'''
with open('learn_data-164-150-100-25000.csv', 'a', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|')
    #spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
    spamwriter.writerow([[333, 666], [1,1]])
'''
