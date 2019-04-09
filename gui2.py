#!/usr/bin/python3
import numpy as np
from tkinter import *
fields = 'Number of \'immediate\' class patients', 'Number of \'delayed\' class patients','Number of Ambulances', \
'Number of Hospitals', 'Distance to Hospitals', 'Servers per triage class per Hospital'

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

class Simulation(object):
    # create the initial conditions for the simulation
    def __init__(self, num_imm, num_del, hospital_distances):
        # initialize the clock and 0 to an empty queue
        self.clock = 0.0

        # set the initial arrival/departure/pickup times
        self.next_ambulance_pickup = 0.0
        self.next_hospital_arrival = BIG
        self.next_immediate_departure = BIG
        self.next_delayed_departure = BIG
        
        # keep track of the efficiency of the system
        self.num_arrivals = 0
        self.num_departures = 0
        self.num_survivals = 0.0
        self.total_wait_time = 0.0
        
        # keep track of all the user entered data
        # hospital distances - generates ambulance travel times
        # number of patients in each class
        self.hospital_distances = hospital_distances
        self.num_imm_patients = num_imm
        self.num_del_patients = num_del
        
        # for running with only 1 hospital we've got the immediate and delayed qs
        self.hospital_imm_q = 0
        self.hospital_del_q = 0
        
        # to handle which q the patient goes to
        self.ambulance_patient = -1
    
    """
    advance_time: we determine the time of the next event, it could be:
    ambulance picks someone up, someone leaves the hospital, someone enters hospital
    """
    def advance_time(self):
        # determine the time of the next event
        time_of_event = min(self.next_hospital_arrival, self.next_immediate_departure, \
                            self.next_delayed_departure, self.next_ambulance_pickup)

        # increment total wait time and update the clock
        self.total_wait_time += (self.num_arrivals- self.num_departures)*(time_of_event - self.clock)
        self.clock = time_of_event

        # handle the arrival event if the next arrival time is 
        # before or the same as the next departure time
        if time_of_event == self.next_hospital_arrival:
            self.handle_arrival_event()
            print("hospital arrival")
        elif time_of_event == self.next_immediate_departure:
            self.handle_immediate_departure_event()
            print("immediate patient departure")
        elif time_of_event == self.next_delayed_departure:
            self.handle_delayed_departure_event()
            print("delayed patient departure")
        else:
            self.handle_ambulance_pickup()
            print("ambulance pickup")
    
    """
    handle_ambulance_pickup: When an ambulance comes to pick up patients
    it should decide between the immediate and delayed patients
    then it should decide the hospital
    """
    def handle_ambulance_pickup(self):
        # set the patient type, defaults to immediate
        self.ambulance_patient = IMMEDIATE
        # decrement whichever queue, for now we'll do imm first then del
        if self.num_imm_patients > 0:
            self.num_imm_patients -= 1
        elif self.num_del_patients >  0:
            self.num_del_patients -= 1
            self.ambulance_patient = DELAYED
        else:
            return
        # handle deciding which hospital to go to, for now we only have one
        # handle the queue at the hospital
        
        # generate next arrival event:
        self.next_hospital_arrival = self.clock + self.generate_ambulance_travel_time()
        self.next_ambulance_pickup = BIG
    
    """
    handle_arrival_event: When we have an arrival event we need to:
    increment the number in the system, and the number of arrivals
    if the new arrival is the only one in the system we schedule its departure
    set the next arrival time to the interarrival RV + the current time
    """
    def handle_arrival_event(self):
        # what is the q length
        q_length = 0
        patient_type = self.ambulance_patient
        # increment the number in the system and of arrivals
        if self.ambulance_patient == IMMEDIATE:
            self.hospital_imm_q += 1
            q_length = self.hospital_imm_q
        else:
            self.hospital_del_q += 1
            q_length = self.hospital_del_q
        self.num_arrivals += 1

        # if this arrival is the only arrival in the system
        # then we schedule its departure
        # hospital departures are currently handled regardless of q type
        if q_length <= 1:
            if patient_type == IMMEDIATE:
                self.next_immediate_departure = self.clock + self.generate_service()
            else:
                self.next_delayed_departure = self.clock + self.generate_service()

        # schedule the next ambulance arrival
        self.next_ambulance_pickup = self.clock + self.generate_ambulance_travel_time()
        self.next_hospital_arrival = BIG

    """
    handle_immediate_departure_event: When a patient leaves the system we want to:
    decrement the number in the system, increment the number of departures
    if the system is not empty then we schedule the next departure
    if the system is empty we set an arbitrarily large next departure time
    """
    def handle_immediate_departure_event(self):
        # increment the number in the system and of arrivals
        self.hospital_imm_q -= 1
        self.num_departures += 1
        
        self.num_survivals += sll_surv_prob(self.clock, IMMEDIATE)
        
        # if this arrival is the only arrival in the system
        # then we schedule its departure
        if self.hospital_imm_q == 0:
            self.next_immediate_departure = BIG
        else:
            self.next_immediate_departure = self.clock + self.generate_service()
    
    def handle_delayed_departure_event(self):
        # increment the number in the system and of arrivals
        self.hospital_del_q -= 1
        self.num_departures += 1
        
        self.num_survivals += sll_surv_prob(self.clock, DELAYED)
        
        # if this arrival is the only arrival in the system
        # then we schedule its departure
        if  self.hospital_del_q > 0:
            self.next_delayed_departure = self.clock + self.generate_service()
        else:
            self.next_delayed_departure = BIG

    def generate_service(self):
        return np.random.exponential(1.)
    
    def generate_ambulance_travel_time(self, distance=1):
        return np.random.lognormal(0.025*distance, 0.01*distance)

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
            hospital_distances = val
        elif(field == fields[5]):
            imm_del_arr = val
    s = Simulation(num_imm, num_del, hospital_distances)
    np.random.seed(0)
    for i in range(64):
        s.advance_time()
    print (s.__dict__)
    return

# shifted log likelihood survival probability
def sll_surv_prob(time, t_class):
    if (t_class == IMMEDIATE):
        beta = sll_pen_imm
    else:
        beta = sll_pen_del
    prob = beta[0]/(1 + (time/beta[1])**beta[2])
    return prob

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
