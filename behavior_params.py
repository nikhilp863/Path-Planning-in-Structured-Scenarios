# Parameters for an agent acting cautiously
class Cautious(object):
    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 10
    tailgate_counter = 0

# Parameters for an agent acting normally
class Normal(object):
    max_speed = 40
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 10
    braking_distance = 5
    tailgate_counter = 0
