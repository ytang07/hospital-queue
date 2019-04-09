#!/usr/bin/python3
import numpy as np
import random
from tkinter import *
fields = 'Number of \'immediate\' class patients', 'Number of \'delayed\' class patients','Number of Ambulances', \
'Number of Hospitals', 'Distance to Hospitals', 'Immediate Servers per Hospital', 'Delayed Servers per Hospital'

EMPTY = -1
IMMEDIATE = 0           # magic number 0
DELAYED = 1             # magic number 1
BIG = 1.0e30            # magic number to set the initial delay so that it happens way after the first arrival
RUNS = 210               # how many times we advance time

# survival probability betas for shifted log logistic
sll_pen_imm = [0.3510, 35.838, 1.9886]          # shifted log logistic for penetrative wounds, immediate class
sll_pen_del = [0.9124, 213.5976, 2.3445]        # shifted log logistic for penetrative wounds, delayed class

num_imm = 20
num_del = 50
num_ams = 5
num_hos = 3
hospital_distances = [5, 10, 20]
imm_arr = [1, 2, 3]
del_arr = [6, 8, 10]

"""
Ambulance Object
Keeps Track of:
Destination = Hospital # or Scene (-1)
Patient Type = IMMEDIATE, DELAYED, or EMPTY
Next Pickup
Next Hospital Arrival
"""

class Ambulance(object):
    def __init__(self):
        self.patient = None
        self.pickup_time = 0.0
        self.dropoff_time = BIG

"""
Hospital Object
Keeps track of:
Distance from Scene
# immediate servers
# delayed servers
# immediate patients
# delayed patients
"""
class Hospital(object):
    def __init__(self, distance, servers_imm, servers_del):
        self.distance = distance
        self.servers_imm = servers_imm
        self.servers_del = servers_del
        self.patients_imm = 0
        self.patients_del = 0

"""
Patient Object
Keeps track of:
Hospital Number
Patient Type
Time arrived at Service
Time departing Service
"""
class Patient(object):
    def __init__(self, patient_type, hospital_number=-1, arrival_time=BIG, departure_time=BIG, _id=0):
        self.patient_type = patient_type
        self.hospital_number = hospital_number
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.survival_probability = 0.0
        self.location = 0               # 0 for not moved, 1 for ambulance, 2 for hospital, 3 for done
        self._id = _id

"""
Simulation Object

Input:
Number of IMMEDIATE patients = n_imm
Number of DELAYED patients = n_del
Number of Ambulances = n_ambs
Number of Hospitals = n_hos
List of Hospital Distances = hos_dists [n_hos distances]
List of Number of Immediate Servers per Hospital = imm_servers [n_hos #imm servers]
List of Number of Delayed Servers per Hospital = del_servers [n_hos #del servers]

Keep track of:
Clock
Array of Ambulances
Array of Hospitals
Array of Patients
Next Pickup Time (minimum of Ambulances[i].next_ambulance_pickup)
Next Hospital Arrival Time (minimum of Ambulances[i].next_ambulance_dropoff)
Next Hospital Departure Time (minimum of Patients[i].departure_time)
Patient Survival Probabilities (sum)
Next Patient to be picked up (number of)

Helper Functions:
Choose Hospital Number (naively random)
Choose Ambulance Number (naively random)
Argmin function

Events:
Advance Time: go to the minimum time event
Pickup Event: Ambulance X picks up Patient Y
Hospital Arrival Event: Ambulance X gives Patient Y to Hospital Z
Patient Departure Event: Patient X departs Hospital Y
"""

class Simulation:
    def __init__(self, n_imm=20, n_del=50, n_ambs=5, n_hos=3, hos_dists=[5,10,20], imm_servers=[1,2,3], del_servers=[6,8,10], seed=12):
        random.seed(seed)
        
        # keeps track of time and our reward probability
        self.clock = 0.0
        self.total_survival_probability = 0.0
        self.patients_picked_up = 0
        
        # keeps track of patients by location
        self.imm_patients = [Patient(IMMEDIATE, _id=i) for i in range(n_imm)]
        self.del_patients = [Patient(DELAYED, _id=-i) for i in range(n_del)]
        self.scene_patients = self.imm_patients + self.del_patients
        self.amb_patients = []
        self.hos_patients = []
        
        # keeps track of ambulances and hospital
        self.ambulances = [Ambulance() for i in range(n_ambs)]
        self.hospitals = [Hospital(hos_dists[i], imm_servers[i], del_servers[i]) for i in range(n_hos)]
        
        # used for determining events
        self.next_ambulance_pickup_time = 0.0                     # min of ambulances[i].next_ambulance_pickup_time
        self.next_ambulance_dropoff_time = BIG           # min of ambulances[i].next_ambulance_dropoff
        self.next_patient_departure_time = BIG          # min of patients[i].departure_time
        
        # used for determining parameters of events
        # self.next_patient_to_pickup = 0
        self.next_patient_to_dropoff = None
        self.next_ambulance_to_pickup = 0
        self.next_ambulance_to_dropoff = None
        self.next_patient_to_depart = None
        
    
    """
    Helper functions
    """
    def _argmin(self, times):
        argmin = 0
        for i in range(len(times)):
            if times[i] < times[argmin]:
                argmin = i
        return argmin
    
    def _argmax(self, times):
        argmax = 0
        for i in range(len(times)):
            if times[i] > times[argmax]:
                argmax = i
        return argmax
    
    def _random_hospital(self, hospital=None):
        if (hospital is None):
            hospital = np.random.randint(len(self.hospitals))
        return hospital
    
    def generate_travel_time(self, distance):
        return 1.5*distance
        # return 60*np.random.lognormal(0.025*distance, 0.01*distance)
    
    # define methods needed to run the simulation
    def generate_next_departure(self, patient_type):
        if patient_type == IMMEDIATE:
            return 90
            #return np.random.exponential(90)
        else:
            return 180
            #return np.random.exponential(180)
        
    # shifted log likelihood survival probability
    def sll_surv_prob(self, time, t_class):
        if (t_class == IMMEDIATE):
            beta = sll_pen_imm
        else:
            beta = sll_pen_del
        prob = beta[0]/(1 + (time/beta[1])**beta[2])
        return prob
    
    def _random_patient(self):
        return np.random.randint(len(self.scene_patients))
    
    """
    patient selection
    myopic approach
    tau_j * r_j * (mu_j/(mu_j + alpha))(beta_j*mu_j/(beta_j*mu_j+alpha))^(x_j+1-beta_j)
    Tau_ij = mean of exponential travel time from location i to location j
    R_j = probability of survival
    B_j = # of servers
    Mu_j = mean service time
    Alpha = expected discount rate? Beta_1 in exponential (-.0207 for IMM, -0.0038 for DEL)
    """
    def _select_patient(self, method="myopic"):
        ptype = -1
        hosnum = 0
        
        if method == "myopic":
            imms = []
            dels = []
            for hospital in self.hospitals:
                tau = self.generate_travel_time(hospital.distance)
                x = self.patients_picked_up
                immr = self.sll_surv_prob(self.clock+tau, IMMEDIATE)
                delr = self.sll_surv_prob(self.clock+tau, DELAYED)
                immb = hospital.servers_imm
                delb = hospital.servers_del
                immmu = self.generate_next_departure(IMMEDIATE)
                delmu = self.generate_next_departure(DELAYED)
                immalpha = -0.0207
                delalpha = -0.0038
                imms.append(tau * immr * (immmu/(immmu+immalpha)) * (immb*immmu/(immb*immmu+immalpha))**(x+1-immb))
                dels.append(tau * delr * (delmu/(delmu+delalpha)) * (delb*delmu/(delb*delmu+delalpha))**(x+1-delb))
            _argmaximm = self._argmax(imms)
            _argmaxdel = self._argmax(dels)
            if imms[_argmaximm] >= dels[_argmaxdel]:
                ptype = IMMEDIATE
                hosnum = _argmaximm
            else:
                ptype = DELAYED
                hosnum = _argmaxdel
        ptypes = [patient.patient_type for patient in self.scene_patients]
        if (ptype in ptypes):
            patient_num = ptypes.index(ptype)
        else:
            patient_num = ptypes.index(1-ptype)
        _patient = self.scene_patients[patient_num]
        _patient.hospital_number = hosnum
        return _patient
    
    """
    advance_time
    controls each step
    """
    
    def advance_time(self):
        # bookkeeping variables
        #self.times = [self.next_ambulance_pickup_time, self.next_ambulance_dropoff_time, self.next_patient_departure_time]
        #self.ambs_tracker = [ambulance.patient_type for ambulance in self.ambulances]
        #self.patient_departures = [patient.departure_time for patient in self.patients]
        #self.patient_arrivals = [patient.arrival_time for patient in self.patients]
        #self.imm_q_tracker = [len(hospital.patients_imm) for hospital in self.hospitals]
        #self.del_q_tracker = [len(hospital.patients_del) for hospital in self.hospitals]
        next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)
        if next_time_step == BIG:
            return
        self.clock = next_time_step
        if next_time_step == self.next_ambulance_pickup_time:
            if len(self.scene_patients) > 0:
                print("pickup event")
                self.pickup_event()
                print([ambulance.pickup_time for ambulance in self.ambulances])
                print([patient.hospital_number for patient in self.amb_patients])
                return
            else:
                next_time_step = min(self.next_ambulance_dropoff_time, self.next_patient_departure_time)
        if next_time_step == self.next_ambulance_dropoff_time:
            if len(self.amb_patients) > 0:
                print("dropoff event")
                print([patient.arrival_time for patient in self.amb_patients])
                print(self.next_patient_to_dropoff)
                self.hospital_arrival_event()
                print(self.next_patient_to_dropoff)
                # print([patient.arrival_time for patient in self.hos_patients])
                # print([ambulance.pickup_time for ambulance in self.ambulances])
                return
            else:
                next_time_step = self.next_patient_departure_time
        if next_time_step == self.next_patient_departure_time:
            if len(self.hos_patients) > 0:
                print("patient departure")
                print([patient.departure_time for patient in self.hos_patients])
                print(self.total_survival_probability)
                self.patient_departure_event()
                return
            else:
                return
    
    """
    pickup_event
    sets the attributes of ambulance number (self.next_ambulance), patient_number (self.next_patient)
    set the ambulance's next pickup time to BIG, and generate hospital arrival time
    updates self.next_ambulance to the ambulance with the minimum pickup time
    updates self.next_patient to the next patient
    updates self.next_ambulance_dropoff_time to that of self.next_ambulance
    """
    def pickup_event(self):
        # grab variables/update patient
        pnum = self._random_patient()
        _patient = self.scene_patients[pnum]
        self.scene_patients[pnum].location = 1
        self.scene_patients[pnum].hospital_number = self._random_hospital()
        _hospital = self.hospitals[self.scene_patients[pnum].hospital_number]
        self.scene_patients[pnum].arrival_time = self.clock + self.generate_travel_time(_hospital.distance)
        
        # update ambulance
        self.ambulances[self.next_ambulance_to_pickup].patient = self.scene_patients[pnum]
        self.ambulances[self.next_ambulance_to_pickup].pickup_time = BIG
        self.ambulances[self.next_ambulance_to_pickup].dropoff_time = self.scene_patients[pnum].arrival_time
        
        # update global variables
        self.next_ambulance_to_dropoff = self._argmin([ambulance.dropoff_time for ambulance in self.ambulances])
        self.next_ambulance_dropoff_time = self.ambulances[self.next_ambulance_to_dropoff].dropoff_time
        
        # update patient arrays
        self.amb_patients.append(self.scene_patients[pnum])
        self.scene_patients.remove(self.scene_patients[pnum])
        
        self.next_patient_to_dropoff = self._argmin([patient.arrival_time for patient in self.amb_patients])
        
        if len(self.scene_patients) > 0:
            self.next_ambulance_to_pickup = self._argmin([ambulance.pickup_time for ambulance in self.ambulances])
            self.next_ambulance_pickup_time = self.ambulances[self.next_ambulance_to_pickup].pickup_time
        else:
            self.next_ambulance_to_pickup = None
            self.next_ambulance_pickup_time = BIG
        
        self.patients_picked_up += 1
        
    def hospital_arrival_event(self):
        # grab patient/hospital/ambulance
        _patient = self.next_patient_to_dropoff
        self.amb_patients[self.next_patient_to_dropoff].location = 2
        _ambulance = self.next_ambulance_to_dropoff
        _hospital = self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number]
        
        # update ambulance
        self.ambulances[_ambulance].patient = None
        self.ambulances[_ambulance].dropoff_time = BIG
        self.ambulances[_ambulance].pickup_time = self.clock + self.generate_travel_time(_hospital.distance)
        
        # update hospital
        if self.amb_patients[self.next_patient_to_dropoff].patient_type == IMMEDIATE:
            self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].patients_imm += 1
            if self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].patients_imm <= self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].servers_imm:
                self.amb_patients[self.next_patient_to_dropoff].departure_time = self.clock + self.generate_next_departure(self.amb_patients[self.next_patient_to_dropoff].patient_type)
                self.amb_patients[self.next_patient_to_dropoff].survival_probability = self.sll_surv_prob(self.amb_patients[self.next_patient_to_dropoff].arrival_time, self.amb_patients[self.next_patient_to_dropoff].patient_type)
            self.total_survival_probability += self.amb_patients[self.next_patient_to_dropoff].survival_probability
        else:
            self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].patients_del += 1
            if self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].patients_del <= self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].servers_del:
                self.amb_patients[self.next_patient_to_dropoff].departure_time = self.clock + self.generate_next_departure(self.amb_patients[self.next_patient_to_dropoff].patient_type)
                self.amb_patients[self.next_patient_to_dropoff].survival_probability = self.sll_surv_prob(self.amb_patients[self.next_patient_to_dropoff].departure_time, self.amb_patients[self.next_patient_to_dropoff].patient_type)
            self.total_survival_probability += self.amb_patients[self.next_patient_to_dropoff].survival_probability
        
        # update global variables
        if len(self.amb_patients) > 0:
            self.next_patient_to_dropoff = self._argmin([patient.arrival_time for patient in self.amb_patients])
            self.next_ambulance_to_dropoff = self._argmin([ambulance.dropoff_time for ambulance in self.ambulances])
            self.next_ambulance_dropoff_time = self.ambulances[_ambulance].dropoff_time
        else:
            self.amb_patients[self.next_patient_to_dropoff] = None
            self.next_ambulance_to_dropoff = -1
            self.next_ambulance_dropoff_time = BIG
        
        # update patient lists
        self.hos_patients.append(self.amb_patients[self.next_patient_to_dropoff])
        self.amb_patients.remove(self.amb_patients[self.next_patient_to_dropoff])
        # get the next ambulance pickup time
        if len(self.scene_patients) > 0:
            self.next_ambulance_to_pickup = self._argmin([ambulance.pickup_time for ambulance in self.ambulances])
            self.next_ambulance_pickup_time = self.ambulances[self.next_ambulance_to_pickup].pickup_time
        
        self.next_patient_to_depart = self._argmin([patient.departure_time for patient in self.hos_patients])
        self.next_patient_departure_time = self.hos_patients[self.next_patient_to_depart].departure_time
    
    def patient_departure_event(self):
        # grab patient and hospital
        _patient = self.hos_patients[self.next_patient_to_depart]
        _hospital = self.hospitals[self.hos_patients[self.next_patient_to_depart].hospital_number]
        
        # update patient
        self.hos_patients[self.next_patient_to_depart].location = 3
        
        # update hospital
        if self.hos_patients[self.next_patient_to_depart].patient_type == IMMEDIATE:
            self.hospitals[self.hos_patients[self.next_patient_to_depart].hospital_number].patients_imm -= 1
        else:
            self.hospitals[self.hos_patients[self.next_patient_to_depart].hospital_number].patients_del -= 1
        
        # update patient list
        self.hos_patients.remove(self.hos_patients[self.next_patient_to_depart])
        
        if len(self.hos_patients) == 0:
            self.next_patient_departure_time = BIG
            return
        
        # select new next patient to depart
        self.next_patient_to_depart = self._argmin([patient.departure_time for patient in self.hos_patients])
        
        for patient in self.hos_patients:
            if patient.patient_type == self.hos_patients[self.next_patient_to_depart].patient_type:
                patient.departure_time = self.clock + self.generate_next_departure(patient.patient_type)
                patient.survival_probability = self.sll_surv_prob(self.clock, patient.patient_type)
        
        # update global variables
        self.total_survival_probability += self.hos_patients[self.next_patient_to_depart].survival_probability

def test(e):
    # assign all variables
    s = Simulation(num_imm, num_del, num_ams, num_hos, hospital_distances, imm_arr, del_arr)
    np.random.seed(0)
    for i in range(RUNS):
        s.advance_time()
    return

def instatiate(e,s):
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
            imm_arr = [int(x.strip()) for x in entry[1].get().split(',')]
        elif(field == fields[6]):
            del_arr = [int(x.strip()) for x in entry[1].get().split(',')]
    s = Simulation(num_imm, num_del, num_ams, num_hos, hospital_distances, imm_arr, del_arr)
    return

def advance(s):
    s.advance_time()
    # call an event
    # update clock
    # make multiple functions - one per event
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
    s = Simulation()
    b1 = Button(root, text='Show',
                command=(lambda e=ents: fetch(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    b2 = Button(root, text = 'Quit', command=root.quit)
    b2.pack(side=LEFT, padx=5, pady=5)
    b3 = Button(root, text = 'Submit', command = (lambda e=ents: instatiate(e,s)))
    b3.pack(side=RIGHT, padx=5, pady=5)
    b4 = Button(root, text = 'Test', command = (lambda e=ents: test(e)))
    b4.pack(side=RIGHT, padx=5, pady=5)
    b5 = Button(root, text = 'advance', command = (lambda e=ents: advance(s)))
    b5.pack(side=RIGHT, padx=5, pady=5)
    root.mainloop()