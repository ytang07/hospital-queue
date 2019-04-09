import numpy as np

BIG = 1.0e30

class Simulation(object):
    # create the initial conditions for the simulation
    def __init__(self):
        # initialize the clock and 0 to an empty queue
        self.clock = 0.0
        self.num_in_system = 0
        
        # set the initial next arrival times
        self.next_arrival_time = self.generate_interarrival()
        self.next_departure_time = BIG
        
        self.num_arrivals = 0
        self.num_departures = 0
        self.total_wait_time = 0.0
        
    def advance_time(self):
        # determine the time of the next event
        time_of_event = min(self.next_arrival_time, self.next_departure_time)
        
        # increment total wait time and update the clock
        self.total_wait_time += self.num_in_system*(time_of_event - self.clock)
        self.clock = time_of_event
        
        # handle the arrival event if the next arrival time is 
        # before or the same as the next departure time
        if self.next_arrival_time <= self.next_departure_time:
            self.handle_arrival_event()
        else:
            self.handle_departure_event()
            
    """
    handle_arrival_event: When we have an arrival event we need to:
    increment the number in the system, and the number of arrivals
    if the new arrival is the only one in the system we schedule its departure
    set the next arrival time to the interarrival RV + the current time
    """
    def handle_arrival_event(self):
        # increment the number in the system and of arrivals
        self.num_in_system += 1
        self.num_arrivals += 1
        
        # if this arrival is the only arrival in the system
        # then we schedule its departure
        if self.num_in_system <= 1:
            self.next_departure_time = self.clock + self.generate_service()
        
        # schedule the next arrival event
        self.next_arrival_time = self.generate_interarrival() + self.clock
        
        
    def handle_departure_event(self):
        self.num_in_system -= 1
        self.num_departures += 1
        
        if self.num_in_system > 0:
            self.next_departure_time = self.clock + self.generate_service()
        else:
            self.next_departure_time = BIG
    
    def generate_interarrival(self):
        return np.random.exponential(1./3)
    
    def generate_service(self):
        return np.random.exponential(1./4)

np.random.seed(0)
s = Simulation()
for i in range(1501):
    s.advance_time()

print (s.__dict__)