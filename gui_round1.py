#!/usr/bin/python3

from tkinter import *
from numpy import *

fields = 'Number of \'immediate\' class patients', 'Number of \'delayed\' class patients','Number of Ambulances', \
'Number of Hospitals', 'Mean Distance and Variance to Hospitals', 'Servers per triage class per Hospital'

IMMEDIATE = 0           # magic number 0
DELAYED = 1             # magic number 1
BIG = 1.0e30            # magic number to set the initial delay so that it happens way after the first arrival

num_imm = 0             # the number of immediate class patients
num_del = 0             # the number of delayed class patients
num_ams = 0             # the number of ambulances
num_hos = 0             # the number of hospitals
mean_var_arr = []       # the mean, variance tuples to each hospital
imm_del_arr = []        # the number of servers for the immediate and delayed classes per hospital as a tuple
patients = []           # array of patients

# survival probability betas for shifted log logistic
sll_pen_imm = [0.3510, 35.838, 1.9886]          # shifted log logistic for penetrative wounds, immediate class
sll_pen_del = [0.9124, 213.5976, 2.3445]        # shifted log logistic for penetrative wounds, delayed class

# make a patient object
class Patient(object):
    def __init__(self, _class):
        self._class = _class
    
    def _does_survive(self, time):
        self._surv = sll_surv_prob(time, self._class)
        return self._surv

# create our simulation class
class Simulate:
    def __init__(self, list_of_patients):
        self.patients = list_of_patients
        self.num_in_q = 0
        
        self.clock = 0.0
        
        self.next_arrival = 0.0
        self.next_departure = BIG
        
        self.num_arrivals = 0
        self.num_departures = 0
        self.total_wait_time = 0.0

    def advance_time(self, time):
        pass
    
    def arrival(self):
        pass
    
    def departure(self):
        pass

# make the list of patients just number of immediate patients followed by the number of delayed patients
def make_patients(num_imm, num_del):
    for i in range(num_imm):
        newp = Patient(IMMEDIATE)
        patients.append(newp)
    for i in range(num_del):
        newp = Patient(DELAYED)
        patients.append(newp)

# instantiates system variables
def instatiate(e):
    # assign all variables
    for entry in e:
        field = entry[0]
        val = int(entry[1].get())
        if (field == fields[0]):
            num_imm = val
        elif (field == fields[1]):
            num_del = val
        elif(field == fields[2]):
            num_ams = val
        elif(field == fields[3]):
            num_hos = val
        elif(field == fields[4]):
            mean_var_list = val
        elif(field == fields[5]):
            imm_del_arr = val
    
    # create list of patients in order
    make_patients(num_imm, num_del)
    time = 0                        # create a global time
    
    # set the clock time to 0
    clock = 0
    
    
    # calculate probability
    total_survival_rate = 0
    while (len(patients)):
        time += travel_times(3,2)
        total_survival_rate += patients[0]._does_survive(time)
        patients.pop()
        time += travel_times(3,2)
    
    print(total_survival_rate)
    return total_survival_rate

# shifted log likelihood survival probability
def sll_surv_prob(time, t_class):
    if (t_class == IMMEDIATE):
        beta = sll_pen_imm
    else:
        beta = sll_pen_del
    prob = beta[0]/(1 + (time/beta[1])**beta[2])
    return prob

def travel_times(mean, var):
    log_mean = log(mean/(sqrt(var/(mean*mean) + 1)))
    log_var = log(1 + var/(mean*mean))
    return (random.lognormal(log_mean, log_var))

def fetch(entries):
    for entry in entries:
        field = entry[0]
        text = entry[1].get()
        print('%s: "%s"' % (field, text))

def makeform(root, fields):
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=35, text=field, anchor='w')
        ent = Entry(row, width=35 )
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))
    return entries

if __name__ == '__main__':
    root = Tk()
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = Button(root, text='Show',
                command=(lambda e=ents: fetch(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    b2 = Button(root, text = 'Quit', command=root.quit)
    b2.pack(side=LEFT, padx=5, pady=5)
    b3 = Button(root, text = 'Submit', command = (lambda e=ents: instatiate(e)))
    b3.pack(side=RIGHT, padx=5, pady=5)
    root.mainloop()
