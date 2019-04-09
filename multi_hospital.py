#!/usr/bin/python3
import numpy as np
import random
from tkinter import *
fields = 'Number of \'immediate\' class patients', 'Number of \'delayed\' class patients','Number of Ambulances', \
'Number of Hospitals', 'Distance to Hospitals', 'Servers per triage class per Hospital'

IMMEDIATE = 0           # magic number 0
DELAYED = 1             # magic number 1
BIG = 1.0e30            # magic number to set the initial delay so that it happens way after the first arrival

# survival probability betas for shifted log logistic
sll_pen_imm = [0.3510, 35.838, 1.9886]          # shifted log logistic for penetrative wounds, immediate class
sll_pen_del = [0.9124, 213.5976, 2.3445]        # shifted log logistic for penetrative wounds, delayed class

"""
Simulation Object should be set up with the following events:
Ambulance X picks up Patient Type Y for Hospital Z
Ambulance X drops off Patient Type Y at Hospital Z
Patient Type X departs Hospital Y

Requires the following objects:
Hospital Object with Immediate Q, Delayed Q
Ambulance Object with Patient type, and Hospital destination

Requires the following abilities:
Generate service times
Generate survival probabilities
Generate travel times
Pick hospitals

Requires the following variables:
Number of Immediate Type Patients
Number of Delayed Type Patients
Number of Ambulances
Number of Hospitals
Distance to each hospital
"""
# define objects
# Ambulance object holds the hospital it needs to go to and the patient type that it carries
class Ambulance(object):
    def __init__(self, num):
        self.num = num
        self.patient_type = -1
        self.dropoff_time = BIG
        self.pickup_time = 0.0
        self.destination = -1

    # pickup a patient - set the patient type, the arrival time the next pickup time, and the destination
    def pickup(self, patient_type, hospital, clock):
        self.patient_type = patient_type
        self.dropoff_time = clock + self.generate_travel_time(hospital.distance)
        self.pickup_time = BIG
        self.destination = hospital.num

    # dropoff a patient - reset the patient type, arrival time, and destination, set next pickup time
    def dropoff(self, hospital, clock):
        self.patient_type = -1
        self.pickup_time = clock + self.generate_travel_time(hospital.distance)
        self.dropoff_time = BIG
        self.destination = -1
    
    def generate_travel_time(self, distance):
        return 60*np.random.lognormal(0.025*distance, 0.01*distance)

# Hospital object holds the immediate and delayed queues for the hospital, 
# the next times of departure for each queue, the distance from the location
class Hospital(object):
    def __init__(self, num, distance):
        self.num = num
        self.immediate_q = 0
        self.delayed_q = 0
        self.distance = distance
        self.next_immediate_departure = BIG
        self.next_delayed_departure = BIG
        self.next_departure = BIG

class Simulation:
    # initial settings
    def __init__(self, number_of_immediate_patients, number_of_delayed_patients,
                number_of_ambulances=1, number_of_hospitals=1, hospital_distances=[1],
                seed=13):
        random.seed(seed)
        # set simulation variables
        self.clock = 0.0
        
        # set events
        self.next_ambulance_pickup = 0.0
        self.next_ambulance_dropoff = BIG
        self.next_patient_departure = BIG
        
        # get our statistical variables
        self.total_hospital_arrivals = 0
        self.total_patient_departures = 0
        self.total_survivors = 0.0
        self.total_pickups = 0
        
        # user specified variables
        self.number_of_immediate_patients = number_of_immediate_patients
        self.number_of_delayed_patients = number_of_delayed_patients
        
        # create our ambulance and hospital objects
        self.ambulances = [Ambulance(i) for i in range(number_of_ambulances)]
        self.hospitals = [Hospital(i,hospital_distances[i]) for i in range(number_of_hospitals)]
        self.ambulance_tracker = [0 for i in range(number_of_ambulances)]
        
        # variables for keeping track of movement
        self.next_ambulance_pickup_number = 0
        self.next_ambulance_dropoff_number = -1
        self.next_hospital_departure_number = -1
    
    # our next time step
    def advance_time(self):
        # determine the time of the next event and advance the clock
        next_event_time = min(self.next_ambulance_pickup, self.next_ambulance_dropoff, self.next_patient_departure)
        self.clock = next_event_time
        
        # handle the event
        if next_event_time == self.next_patient_departure:
            print("patient departs from hospital", self.next_hospital_departure_number)
            self.handle_next_patient_departure(self.hospitals[self.next_hospital_departure_number])
        elif next_event_time == self.next_ambulance_dropoff:
            print("patient dropped off at hospital")
            self.handle_next_hospital_arrival(self.next_ambulance_dropoff_number)
        elif next_event_time == self.next_ambulance_pickup:
            print("ambulance pickup")
            self.handle_next_ambulance_pickup()
        
        print(self.__dict__)
        
    
    # helper functions
    # primitive assign patient type function
    def assign_patient_type(self):
        if self.number_of_immediate_patients > 0:
            return IMMEDIATE
        return DELAYED
    
    # primitive assign hospital function
    def assign_hospital(self):
        return np.random.randint(len(self.hospitals))
    
    # np.argmin doesn't seem to work for me
    def min_dropoff(self):
        times = [ambulance.dropoff_time for ambulance in self.ambulances]
        argmin = 0
        for i in range(len(times)):
            if times[i] < times[argmin]:
                argmin = i
        return argmin
    
    def min_depart(self):
        times = [hospital.next_departure for hospital in self.hospitals]
        argmin = 0
        for i in range(len(times)):
            if times[i] < times[argmin]:
                argmin = i
        return argmin
    
    # define events
    """
    handling the next ambulance pickup should consist of: incrementing the number of total pickups,
    assigning an ambulance with it's patient type and hospital destination, generating the ambulance's
    arrival time and determining the next ambulance arrival at a hospital
    """
    def handle_next_ambulance_pickup(self):
        # pick the ambulance
        ambulance_number = self.next_ambulance_pickup_number
        self.ambulance_tracker[ambulance_number] = 1
        
        # incrememnt number of pickups
        self.total_pickups += 1
        
        # assign the patient type and the destination/hospital
        patient_type = self.assign_patient_type()
        hospital = self.hospitals[self.assign_hospital()]
        
        # decrement patient types at the waiting zone
        if patient_type == IMMEDIATE:
            self.number_of_immediate_patients -= 1
        else:
            self.number_of_delayed_patients -= 1
        
        # pickup the patient
        self.ambulances[ambulance_number].pickup(patient_type, hospital, self.clock)
        
        # determine which ambulance will drop off next and when it will drop off
        self.next_ambulance_dropoff_number = np.argmin(self.ambulances[i].dropoff_time for i in range(len(self.ambulances)))
        self.next_ambulance_dropoff = min(ambulance.dropoff_time for ambulance in self.ambulances)
        
        self.next_ambulance_pickup = min(ambulance.pickup_time for ambulance in self.ambulances)
        if 0 in self.ambulance_tracker:
            self.next_ambulance_pickup_number = self.ambulance_tracker.index(0)
    
    """
    handling the next hospital dropoff requires knowledge of the next arriving ambulance number
    we need to increment the number of hospital arrivals, add the patient to a q
    assign the next departure time only if the q is size 1, then update the ambulance
    """
    def handle_next_hospital_arrival(self, ambulance_number):
        # increment total arrivals
        self.total_hospital_arrivals += 1
        
        # clear the ambulance for use
        self.ambulance_tracker[ambulance_number] = 0
        
        # get the hospital number and the patient type from the ambulance
        hospital_number = self.ambulances[ambulance_number].destination
        patient_type = self.ambulances[ambulance_number].patient_type
        
        # handle the patient by type
        if patient_type == IMMEDIATE:
            self.hospitals[hospital_number].immediate_q += 1
            if self.hospitals[hospital_number].immediate_q <= 1:
                self.hospitals[hospital_number].next_immediate_departure = self.clock + self.generate_next_departure(patient_type)
        else:
            self.hospitals[hospital_number].delayed_q += 1
            if self.hospitals[hospital_number].delayed_q <= 1:
                self.hospitals[hospital_number].next_delayed_departure = self.clock + self.generate_next_departure(patient_type)
        
        # find the next closest departure time of any patient in any hospital
        self.hospitals[hospital_number].next_departure = min(self.hospitals[hospital_number].next_delayed_departure, self.hospitals[hospital_number].next_immediate_departure)
        self.next_hospital_departure_number = np.argmin(h.next_departure for h in self.hospitals)
        self.next_patient_departure = self.hospitals[self.next_hospital_departure_number].next_departure
        
        self.ambulances[ambulance_number].dropoff(self.hospitals[hospital_number], self.clock)
        self.next_ambulance_dropoff_number = self.min_dropoff()
        
        self.next_ambulance_dropoff = min(ambulance.dropoff_time for ambulance in self.ambulances)
        self.next_ambulance_pickup = min(ambulance.pickup_time for ambulance in self.ambulances)
        
    """
    handles hospital departure event, we need to determine the appropriate q,
    decrement the q, increment number of departures, update survival probability
    set the next departure time if the q is nonempty
    """
    def handle_next_patient_departure(self, hospital):
        self.total_patient_departures += 1
        if hospital.next_immediate_departure < hospital.next_delayed_departure:
            hospital.immediate_q -= 1
            self.total_survivors += self.sll_surv_prob(hospital.next_immediate_departure, IMMEDIATE)
            if hospital.immediate_q >= 1:
                hospital.next_immediate_departure = self.clock + self.generate_next_departure(IMMEDIATE)
            else:
                hospital.next_immediate_departure = BIG
        else:
            hospital.delayed_q -= 1
            self.total_survivors += self.sll_surv_prob(hospital.next_delayed_departure, DELAYED)
            if hospital.delayed_q >= 1:
                hospital.next_delayed_departure = self.clock + self.generate_next_departure(DELAYED)
            else:
                hospital.next_delayed_departure = BIG
        
        hospital.next_departure = min(hospital.next_delayed_departure, hospital.next_immediate_departure)
        self.next_hospital_departure_number = self.min_depart()
        self.next_patient_departure = self.hospitals[self.next_hospital_departure_number].next_departure
    
    # define methods needed to run the simulation
    def generate_next_departure(self, patient_type):
        if patient_type == IMMEDIATE:
            return np.random.exponential(90)
        else:
            return np.random.exponential(180)
    
    # shifted log likelihood survival probability
    def sll_surv_prob(self, time, t_class):
        if (t_class == IMMEDIATE):
            beta = sll_pen_imm
        else:
            beta = sll_pen_del
        prob = beta[0]/(1 + (time/beta[1])**beta[2])
        return prob

def instatiate(e):
    # assign all variables
    for entry in e:
        field = entry[0]
        if (field == fields[0]):
            num_imm = int(entry[1].get())
        elif (field == fields[1]):
            num_del = int(entry[1].get())
        elif(field == fields[2]):
            num_ams = int(entry[1].get())
        elif(field == fields[3]):
            num_hos = int(entry[1].get())
        elif(field == fields[4]):
            hospital_distances = [float(x.strip()) for x in entry[1].get().split(',')]
        elif(field == fields[5]):
            imm_del_arr = [int(x.strip()) for x in entry[1].get().split(',')]
    s = Simulation(num_imm, num_del, num_ams, num_hos, hospital_distances)
    np.random.seed(0)
    for i in range(64):
        s.advance_time()
    return


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