#!/usr/bin/python3
import numpy as np
import random
from tkinter import *
import csv
fields = 'Number of \'immediate\' class patients', 'Number of \'delayed\' class patients','Number of Ambulances', \
'Number of Hospitals', 'Distance to Hospitals', 'Immediate Servers per Hospital', 'Delayed Servers per Hospital'

EMPTY = -1
IMMEDIATE = 0           # magic number 0
DELAYED = 1             # magic number 1
BIG = 1.0e30            # magic number to set the initial delay so that it happens way after the first arrival

# survival probability betas for shifted log logistic
sll_pen_imm = [0.3510, 35.838, 1.9886]          # shifted log logistic for penetrative wounds, immediate class
sll_pen_del = [0.9124, 213.5976, 2.3445]        # shifted log logistic for penetrative wounds, delayed class
immalpha = -0.0207
delalpha = -0.0038

# global for button use
SELECTED_PATIENT = 0
SELECTED_HOSPITAL = 0


# mandalay bay test
# immediate patients: uniform(10-40)% of uniform(200-250) total patients
# hospital distances: 5.59, 4.24, 6.95
# hospital imm servers: 6, 5, 0
# hospital del servers: 15, 12, 8
# 30 ambulances

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
    def __init__(self, patient_type, hospital_number=-1, arrival_time=BIG, departure_time=BIG):
        self.patient_type = patient_type
        self.hospital_number = hospital_number
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.survival_probability = 0.0
        self.location = 0               # 0 for not moved, 1 for ambulance, 2 for hospital, 3 for done

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

# n_imm=20, n_del=50, n_ambs=2, n_hos=3, hos_dists=[5,10,20], imm_servers=[1,2,3], del_servers=[6,8,10], selection="random", seed=12

class Simulation:
    def __init__(self, *args):
        self.clock = 0.0
        
    def true_init(self, n_imm=20, n_del=50, n_ambs=2, n_hos=3, hos_dists=[5,10,20], imm_servers=[1,2,3], del_servers=[6,8,10], selection="random", seed=12):
        random.seed(seed)
        
        # controls selection of patients
        self._select = False
        self.imm_picked = 0
        self.del_picked = 0
        #self._selected_patient = 0
        #self._selected_hospital = 0
        
        # keeps track of time and our reward probability
        self.clock = 0.0
        self.total_survival_probability = 0.0
        self.served = 0
        self.limit = n_imm + n_del
        self.patients_picked_up = 0
        self.selection = selection
        
        # keeps track of patients by location
        self.imm_patients = [Patient(IMMEDIATE) for i in range(n_imm)]
        self.del_patients = [Patient(DELAYED) for i in range(n_del)]
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
        self.next_time_step = 0.0
        
        # used for determining parameters of events
        # self.next_patient_to_pickup = 0
        self.next_patient_to_dropoff = None
        self.next_ambulance_to_pickup = 0
        self.next_ambulance_to_dropoff = None
        self.next_patient_to_depart = None
    
    
    """
    pickup patient selection
    """
    def is_select(self):
        self._select = True
        return
    
    def patient_select(self, patient_type):
        if patient_type == DELAYED:
            self._selected_patient = len(self.scene_patients)-1
        elif patient_type == IMMEDIATE:
            self._selected_patient = 0
    
    def hospital_select(self, hospital_number):
        self._selected_hospital = hospital_number
    
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
        # return 1.5*distance
        return 60*np.random.lognormal(0.025*distance, 0.01*distance)
    
    # define methods needed to run the simulation
    def generate_next_departure(self, patient_type):
        if patient_type == IMMEDIATE:
            #return 90
            return np.random.exponential(90)
        else:
            #return 180
            return np.random.exponential(180)
        
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
    def _myopic_reward(self, patient_type, hospital):
        # separate by whether patient is immediate or delayed
        tau = 60*0.025*self.hospitals[hospital].distance
        if patient_type == IMMEDIATE:
            x = self.hospitals[hospital].patients_imm
            immr = self.sll_surv_prob(self.clock+tau, IMMEDIATE)
            immb = self.hospitals[hospital].servers_imm
            immmu = self.clock + self.generate_next_departure(IMMEDIATE)
            return (tau * immr * (immmu/(immmu+immalpha)) * (immb*immmu/(immb*immmu+immalpha))**(x+1-immb))
        else:
            x = self.hospitals[hospital].patients_imm
            delr = self.sll_surv_prob(self.clock+tau, DELAYED)
            delb = self.hospitals[hospital].servers_del
            delmu = self.clock + self.generate_next_departure(DELAYED)
            return (tau * delr * (delmu/(delmu+delalpha)) * (delb*delmu/(delb*delmu+delalpha))**(x+1-delb))
        
    def _myopic(self):
        # create pair patient_num, hospital by reward
        _opt = [0,0]
        _del = None
        _imm = None
        if DELAYED in [patient.patient_type for patient in self.scene_patients]:
            _del = [patient.patient_type for patient in self.scene_patients].index(DELAYED)
        if IMMEDIATE in [patient.patient_type for patient in self.scene_patients]:
            _imm = [patient.patient_type for patient in self.scene_patients].index(IMMEDIATE)
        if (_del is None):
            assert _imm is not None
        for hospital in range(len(self.hospitals)):
            if self._myopic_reward(IMMEDIATE, _opt[1]) < self._myopic_reward(DELAYED, hospital) and _del is not None:
                _opt[0] = _del
                _opt[1] = hospital
            elif _imm is not None:
                _opt[0] = _imm
                _opt[1] = hospital
        self.scene_patients[_opt[0]].hospital_number = _opt[1]
        return(_opt[0])
    
    """
    advance_time
    controls each step
    """
    
    def advance_time(self):
        # bookkeeping variables
        if (self.served == self.limit):
            return self.total_survival_probability
        #if next_time_step == BIG:
            #print(self.__dict__)
            #return self.total_survival_probability
        self.clock = self.next_time_step
        #print(str(self.next_ambulance_pickup_time), str(self.next_ambulance_dropoff_time), str(self.next_patient_departure_time))
        if self.next_time_step == self.next_ambulance_pickup_time:
            if len(self.scene_patients) > 0:
                if self._select:
                    print("There are ",str(len(self.imm_patients)-self.imm_picked), " IMMEDIATE triage class patients left at the scene")
                    immatscene = "There are "+ str(len(self.imm_patients)-self.imm_picked) + " IMMEDIATE triage class patients left at the scene"
                    print("There are ",str(len(self.del_patients)-self.del_picked), " DELAYED triage class patients left at the scene")
                    delatscene = "There are " + str(len(self.del_patients)-self.del_picked) + " DELAYED triage class patients left at the scene"
                    info = Tk()
                    info.winfo_toplevel().title("Relevant Information")
                    l1 = Label(info, text=immatscene)
                    l1.grid(row=0, column=0)
                    l2 = Label(info, text=delatscene)
                    l2.grid(row=1, column=0)
                    for i in range(len(self.ambulances)):
                        if self.ambulances[i].pickup_time == BIG:
                            if self.ambulances[i].patient.patient_type == 0:
                                _type = "an IMMEDIATE"
                            else:
                                _type = "a DELAYED"
                            print("Ambulance ", str(i), " is taking ", _type, 
                                  " type patient to Hospital number ", str(self.ambulances[i].patient.hospital_number))
                            #_text = "Ambulance "+ str(i), " is taking ", _type + " type patient to Hospital number " + str(self.ambulances[i].patient.hospital_number)
                    for i in range(len(self.hospitals)):
                        hos = self.hospitals[i]
                        if hos.patients_imm > hos.servers_imm:
                            print("IMMEDIATE Queue Size in Hospital ", str(i), ": ", str(hos.patients_imm - hos.servers_imm))
                        else:
                            print("Free IMMEDIATE Servers in Hospital ", str(i),": ", str(hos.servers_imm - hos.patients_imm))
                        
                        if hos.patients_del > hos.servers_del:
                            print("DELAYED Queue Size in Hospital ", str(i), ": ", str(hos.patients_del - hos.servers_del))
                        else:
                            print("Free DELAYED Servers in Hospital ", str(i), ": ", str(hos.servers_del - hos.patients_del))
                    select = Tk()
                    select.winfo_toplevel().title("Select Patient to pickup")
                    l1 = Label(select, text="Choose Patient Type")
                    l1.grid(row=0, column=0)
                    b1 = Button(select, text='IMMEDIATE',command=(lambda e=IMMEDIATE: self.change_patient(e)))
                    b1.grid(row=1, column=0)
                    b2 = Button(select, text='DELAYED', command=(lambda e=DELAYED: self.change_patient(e)))
                    b2.grid(row=1, column=1)
                    e1 = Entry(select)
                    e1.grid(row=2, column=1)
                    b3 = Button(select, text="Select Hospital", command=(lambda e=e1: self.change_hospital(e)))
                    b3.grid(row=2, column=0)
                    b4 = Button(select, text="Select", command=(lambda select=select, info=info: self._close(select, info)))
                    b4.grid(row=3, column=1)
                else:
                    self.pickup_event()
                    self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)
                #print([ambulance.pickup_time for ambulance in self.ambulances])
                #print([patient.hospital_number for patient in self.amb_patients])
                return
            else:
                self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_patient_departure_time)
        if self.next_time_step == self.next_ambulance_dropoff_time:
            if len(self.amb_patients) > 0:
                print("dropoff event")
                #print([patient.arrival_time for patient in self.amb_patients])
                #print(self.next_patient_to_dropoff)
                self.hospital_arrival_event()
                #print(self.next_patient_to_dropoff)
                # print([patient.arrival_time for patient in self.hos_patients])
                # print([ambulance.pickup_time for ambulance in self.ambulances])
                self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)
                return
            else:
                self.next_time_step = self.next_patient_departure_time
        if self.next_time_step == self.next_patient_departure_time:
            if len(self.hos_patients) > 0:
                print("patient departure")
                #print([patient.departure_time for patient in self.hos_patients])
                #print(self.total_survival_probability, self.served)
                self.patient_departure_event()
                #self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)
                return
            else:
                return
    
    def _close(self, select, info):
        #print(str(SELECTED_PATIENT), ' ', str(SELECTED_HOSPITAL))
        self.pickup_event()
        info.destroy()
        select.destroy()
    
    def change_patient(self, patient_type):
        self.patient_select(patient_type)
        if patient_type == IMMEDIATE:
            print("IMMEDIATE type patient selected")
        elif patient_type == DELAYED:
            print("DELAYED type patient selected")
    
    def change_hospital(self, e):
        hospital_number = int(e.get())
        self.hospital_select(hospital_number)
        print("Patient will be moved to Hospital ", str(hospital_number))
    
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
        print("pickup event")
        if self._select:
            pnum = self._selected_patient
            self.scene_patients[pnum].hospital_number = self._selected_hospital
        elif self.selection == "random":
            pnum = self._random_patient()
            self.scene_patients[pnum].hospital_number = self._random_hospital()
        elif self.selection == "myopic":
            pnum = self._myopic()
        elif self.selection == "last":
            pnum = len(self.scene_patients)-1
            self.scene_patients[pnum].hospital_number = self._random_hospital()
        elif self.selection == "first":
            pnum = 0
            self.scene_patients[pnum].hospital_number = self._random_hospital()
        
        # increment correct patient type picked up from scene
        if self.scene_patients[pnum].patient_type == IMMEDIATE:
            self.imm_picked += 1
        else:
            self.del_picked += 1
        
        #print(pnum)
        self.scene_patients[pnum].location = 1
        _hospital = self.hospitals[self.scene_patients[pnum].hospital_number]
        print(self.scene_patients[pnum].hospital_number)
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
        self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)
        
    def hospital_arrival_event(self):
        # grab patient/hospital/ambulance
        #_patient = self.next_patient_to_dropoff
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
                self.amb_patients[self.next_patient_to_dropoff].survival_probability = self.sll_surv_prob(self.clock, self.amb_patients[self.next_patient_to_dropoff].patient_type)
            self.total_survival_probability += self.amb_patients[self.next_patient_to_dropoff].survival_probability
        else:
            self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].patients_del += 1
            if self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].patients_del <= self.hospitals[self.amb_patients[self.next_patient_to_dropoff].hospital_number].servers_del:
                self.amb_patients[self.next_patient_to_dropoff].departure_time = self.clock + self.generate_next_departure(self.amb_patients[self.next_patient_to_dropoff].patient_type)
                self.amb_patients[self.next_patient_to_dropoff].survival_probability = self.sll_surv_prob(self.clock, self.amb_patients[self.next_patient_to_dropoff].patient_type)
            self.total_survival_probability += self.amb_patients[self.next_patient_to_dropoff].survival_probability
        
        # update patient lists
        self.hos_patients.append(self.amb_patients[self.next_patient_to_dropoff])
        self.amb_patients.remove(self.amb_patients[self.next_patient_to_dropoff])
        # update global variables
        if len(self.amb_patients) > 0:
            self.next_patient_to_dropoff = self._argmin([patient.arrival_time for patient in self.amb_patients])
            self.next_ambulance_to_dropoff = self._argmin([ambulance.dropoff_time for ambulance in self.ambulances])
            self.next_ambulance_dropoff_time = self.ambulances[_ambulance].dropoff_time
        else:
            self.next_patient_to_dropoff = -1
            self.next_ambulance_to_dropoff = -1
            self.next_ambulance_dropoff_time = BIG
        
        # get the next ambulance pickup time
        if len(self.scene_patients) > 0:
            self.next_ambulance_to_pickup = self._argmin([ambulance.pickup_time for ambulance in self.ambulances])
            self.next_ambulance_pickup_time = self.ambulances[self.next_ambulance_to_pickup].pickup_time
        
        self.next_patient_to_depart = self._argmin([patient.departure_time for patient in self.hos_patients])
        self.next_patient_departure_time = self.hos_patients[self.next_patient_to_depart].departure_time
        
        print(self.total_survival_probability)
        self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)
    
    def patient_departure_event(self):
        self.served += 1
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
                if (self.hos_patients[self.next_patient_to_depart].patient_type == IMMEDIATE):
                    self.hospitals[self.hos_patients[self.next_patient_to_depart].hospital_number].patients_imm += 1
                else:
                    self.hospitals[self.hos_patients[self.next_patient_to_depart].hospital_number].patients_del -= 1
                break
        
        # update global variables
        self.total_survival_probability += self.hos_patients[self.next_patient_to_depart].survival_probability
        print(self.total_survival_probability)
        self.next_time_step = min(self.next_ambulance_dropoff_time, self.next_ambulance_pickup_time, self.next_patient_departure_time)

def test():
    # assign all variables
    # mandalay bay test
    # immediate patients: uniform(10-40)% of uniform(200-250) total patients
    # hospital distances: 5.59, 4.24, 6.95
    # hospital imm servers: 6, 5, 0
    # hospital del servers: 15, 12, 8
    # 30 ambulances
    num_ams = 30
    num_hos = 3
    hospital_distances = [5.59, 4.24, 6.95]
    imm_arr = [6, 5, 0]
    del_arr = [15, 12, 8]
    obs = []
    for i in range(100):
        np.random.seed(i)
        total_patients = np.random.randint(200, 251)
        perc_imm = np.random.uniform(.1, .4)
        num_imm = int(round(perc_imm*total_patients))
        num_del = total_patients - num_imm
        selection = "random"
        #selection = "first"
        #selection = "last"
        #selection = "myopic"
        s=Simulation()
        s.true_init(num_imm, num_del, num_ams, num_hos, hospital_distances, imm_arr, del_arr, selection)
        for i in range(total_patients*3):
            s.advance_time()
            tup = [s.total_survival_probability, s.served, num_imm, num_del]
        print(tup)
        obs.append(tup)
    if (selection == "myopic"):
        with open('myopic.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(obs)
        writeFile.close()
    elif (selection == "random"):
        with open('random.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(obs)
        writeFile.close()
    elif (selection == "first"):
        with open('first.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(obs)
        writeFile.close()
    elif (selection == "last"):
        with open('last.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(obs)
        writeFile.close()
    return

def instantiate(e, SIM):
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
    SIM = SIM.true_init(num_imm, num_del, num_ams, num_hos, hospital_distances, imm_arr, del_arr)
    return

#def self_select(entries, s, ent_patients, ent_hospitals):
    #s._selected_patient = int(entries[0][1].get())
    #s._selected_hospital = int(entries[1][1].get())
    #print("Patient " + entries[0][1].get() + " picked up to go to Hospital " + entries[1][1].get())
    #s.advance_time()
    #ent_hospitals.delete(0, END)
    #ent_patients.delete(0, END)
    #while s.next_ambulance_pickup_time > s.next_ambulance_dropoff_time and s.next_ambulance_pickup_time > s.next_patient_departure_time:
        #s.advance_time()
    #if s._next == s.next_ambulance_pickup_time and s._next != 1e30:
        #print("Pickup Selection")
    #return

def advance(s):
    # Show queue at hospital
    # Show number of patients on the way
    # Show Free Servers at hospital
    # Select Immediate/Delayed by type and which hospital
    # decision made at pickup_time
    # Show number of Immediate/Delayed Patients at scene
    # show state of ambulances
    
    """
    Pops up window allowing selection of patient type and hospital number
    Print Immediate/Delayed Patients at scene
    Print Ambulance states/dict
    Print Hospital free servers
    """
    
    #def _close(select, s):
        ##print(str(SELECTED_PATIENT), ' ', str(SELECTED_HOSPITAL))
        #select.destroy()
        #print("pickup event")
    
    #def change_patient(s, patient_type):
        #s.patient_select(patient_type)
        #if patient_type == IMMEDIATE:
            #print("IMMEDIATE type patient selected")
        #elif patient_type == DELAYED:
            #print("DELAYED type patient selected")
    
    #def change_hospital(s, e):
        #hospital_number = e.get()
        #s.hospital_select(hospital_number)
        #print("Patient will be moved to Hospital ", str(SELECTED_HOSPITAL))
    
    s.is_select()
    #if s.next_time_step == s.next_ambulance_pickup_time:
        #select = Tk()
        #select.winfo_toplevel().title("Select Patient to pickup")
        #l1 = Label(select, text="Choose Patient Type")
        #l1.grid(row=0, column=0)
        #b1 = Button(select, text='IMMEDIATE',command=(lambda s=s, e=IMMEDIATE: change_patient(s,e)))
        #b1.grid(row=1, column=0)
        #b2 = Button(select, text='DELAYED', command=(lambda s=s, e=DELAYED: change_patient(s,e)))
        #b2.grid(row=1, column=1)
        #e1 = Entry(select)
        #e1.grid(row=2, column=1)
        #b3 = Button(select, text="Select Hospital", command=(lambda s=s, e=e1: change_hospital(s,e)))
        #b3.grid(row=2, column=0)
        #b4 = Button(select, text="Select", command=(lambda select=select, s=s: _close(select, s)))
        #b4.grid(row=3, column=1)
        #s.advance_time()
    #else:
    s.advance_time()
    return

def fetch(entries):
    #for key in entries.keys():
        #print(key, entries[key])
    for entry in entries:
        field = entry[0]
        text = entry[1].get()
        print('%s: "%s"' % (field, text))

def show_dict(entries):
    for key in entries.keys():
        print(key, entries[key])

def makeform(root, fields):
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=35, text=field, anchor='w')
        ent = Entry(row, width=35)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))
    return entries

if __name__ == '__main__':
    root = Tk()
    root.winfo_toplevel().title("Emergency Room Simulator")
    ents = makeform(root, fields)
    SIM = Simulation()#n_imm=20, n_del=12, n_ambs=1, n_hos=3, hos_dists=[1, 3, 6], imm_servers=[2, 2, 1], del_servers=[3, 2, 4])
    root.bind('<Return>', (lambda event, e=SIM.__dict__: show_dict(e)))
    b1 = Button(root, text='Show',command=(lambda e=ents: fetch(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    b2 = Button(root, text = 'Quit', command=root.quit)
    b2.pack(side=LEFT, padx=5, pady=5)
    b3 = Button(root, text = 'Submit', command = (lambda e=ents: instantiate(e, SIM)))
    b3.pack(side=RIGHT, padx=5, pady=5)
    b4 = Button(root, text = 'Test', command = (lambda: test()))
    b4.pack(side=RIGHT, padx=5, pady=5)
    b5 = Button(root, text = 'Advance', command = (lambda s=SIM: advance(s)))
    b5.pack(side=RIGHT, padx=5, pady=5)
    root.mainloop()

