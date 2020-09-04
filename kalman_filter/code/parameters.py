def thrust(time_step):
    if time_step >= 0 and time_step < 5:
        return 500
    elif time_step >= 25 and time_step < 30:
        return -500
    else:
        return 0

VEHICLE_MASS = 100 # kg
LINEAR_DRAG_COEFFICIENT = 20 # N*s/m
MEASUREMENT_NOISE_COVARIANCE = 0.00001 # m^2
SAMPLE_PERIOD = 0.05 # Seconds
PROCESS_NOISE_COVARIANCE_VELOCITY = 0.01 # m^2/s^2 
PROCESS_NOISE_COVARIANCE_POSITION = 0.0001 # m^2