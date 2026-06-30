import numpy as np
import matplotlib.pyplot as plt

class Nucleon:
    def __init__(self, x_pos):
        self.x = x_pos
        self.zeta = 1.0  # Starts as upright Proton
        self.v_rad = 1.0 
        self.radius = 0.5 # Physical boundary of the vortex
        
    def get_impedance(self, other_x):
        # The distance between the centers
        dist = abs(self.x - other_x)
        # The collision distance (r_a + r_b)
        contact_dist = self.radius * 2
        
        # Emerging Impedance 'a':
        # If dist > contact_dist, 'a' drops (attraction/bridge)
        # If dist <= contact_dist, 'a' spikes (repulsion/collision)
        # We use a stiff inverse-power law for the collision
        if dist < contact_dist:
            return 100 / (dist - (contact_dist * 0.9))**2 # The Spike
        else:
            return -5 / (dist**2) # The Bridge (Attraction)

    def update(self, other_x):
        # 1. Calculate the emergent impedance 'a'
        a = self.get_impedance(other_x)
        
        # 2. Response: Torque (dZeta/dt) is driven by 'a'
        # If 'a' is positive (collision), torque pushes zeta to 0 (Neutron)
        # If 'a' is negative (bridge), zeta stays 1 (Proton)
        torque = a * 0.001 
        self.zeta = np.clip(self.zeta - torque, 0.0, 1.0)
        
        # Pulse is masked by the tilt (Zeta)
        self.v_rad = self.zeta

# --- Simulation ---
p1 = Nucleon(0)
p2 = Nucleon(0.6) # Very close, will trigger collision

history = {"p2_zeta": [], "a_spike": []}

for t in range(500):
    p2.update(p1.x)
    history["p2_zeta"].append(p2.zeta)
    history["a_spike"].append(p2.get_impedance(p1.x))

# --- Visualization ---
fig, ax1 = plt.subplots()
ax1.plot(history["p2_zeta"], color='blue', label='Zeta (Tilt State)')
ax1.set_ylabel("Zeta")
ax2 = ax1.twinx()
ax2.plot(history["a_spike"], color='red', label='Emergent Impedance (a)')
ax2.set_ylabel("Impedance 'a'")
plt.show()