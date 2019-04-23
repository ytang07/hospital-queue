import numpy as np
from matplotlib import pyplot as plt
import csv

myopic_surv_rates = []
myopic_total_patients = []
myopic_immediate_patients = []
myopic_delayed_patients = []
with open('myopic.csv') as infile:
    read = csv.reader(infile, delimiter = ',')
    for row in read:
        if len(row) > 0:
            myopic_surv_rates.append(float(row[0]))
            myopic_total_patients.append(int(row[1]))
            myopic_immediate_patients.append(row[2])
            myopic_delayed_patients.append(row[3])

random_surv_rates = []
random_total_patients = []
random_immediate_patients = []
random_delayed_patients = []
with open('random.csv') as infile:
    read = csv.reader(infile, delimiter = ',')
    for row in read:
        if len(row) > 0:
            random_surv_rates.append(float(row[0]))
            random_total_patients.append(int(row[1]))
            random_immediate_patients.append(row[2])
            random_delayed_patients.append(row[3])
            

fig = plt.figure()
myopic = fig.add_subplot(121)
random = fig.add_subplot(122)
myopic.scatter(myopic_total_patients, myopic_surv_rates, label="myopic selection")
myopic.set_xlim(left=200, right=250)
myopic.set_ylim(bottom=45, top=100)
myopic.set_xticks(np.arange(200, 250, step=5))
myopic.set_yticks(np.arange(50, 100, step=5))
myopic.set_title("Myopic Selection")

random.scatter(random_total_patients, random_surv_rates)
random.set_xlim(200, 250)
random.set_ylim(45, 100)
random.set_xticks(range(200, 250, 5))
random.set_yticks(range(45, 100, 5))
random.set_title("Random Selection")
plt.show()