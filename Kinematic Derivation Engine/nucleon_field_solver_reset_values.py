# Change these values inside your existing function to optimize the timeline:

system_A = VortexSystem("Proton_A", position=0.0, initial_velocity=0.005) # Boosted speed
system_B = VortexSystem("Proton_B", position=0.42, initial_velocity=-0.005) # Closer start

# Inside the loop:
torque_A = pulse_clash * 0.45  # Significantly increased from 0.08
torque_B = pulse_clash * 0.45  # Significantly increased from 0.08

spatial_force = (pulse_clash * 0.15) - (0.003 / (distance_AB**2)) # Punchier gradients