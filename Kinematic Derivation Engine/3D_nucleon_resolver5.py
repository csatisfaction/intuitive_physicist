import numpy as np
import matplotlib.pyplot as plt

class Nucleon:
    def __init__(self, x_pos, velocity):
        self.x = x_pos
        self.v = velocity
        self.zeta = 1.0 # Start as Proton
        
    def get_forces(self, other):
        dist = abs(self.x - other.x)
        # Softer exponential potential for collision
        # No longer an infinite spike
        return 0.5 * np.exp(-(dist - 0.2) * 10) 

# --- Simulation Setup ---
p1 = Nucleon(0, 0)
p2 = Nucleon(0.6, -0.005) # Slower approach for better resolution

history = {"zeta": [], "dist": [], "a": []}

for t in range(500):
    force = p2.get_forces(p1)
    
    # 1. Physics: Integration
    p2.v += force * 0.001
    p2.x += p2.v
    
    # 2. Dynamics: Zeta with Damping (The "Snap" Curve)
    # Torque sensitivity scaled down significantly
    torque = force * 0.005 
    
    # Differential equation: Zeta moves toward 0 when force is high
    # Moves toward 1 when force is low (relaxation)
    p2.zeta += (-torque * p2.zeta) + (0.0001 * (1.0 - p2.zeta))
    p2.zeta = np.clip(p2.zeta, 0.0, 1.0)
    
    # Track
    history["zeta"].append(p2.zeta)
    history["a"].append(force)

# --- Visualization ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(history["zeta"], label='Zeta (Tilt)', color='blue', linewidth=2)
ax1.set_ylabel("Coupling Constant ($\zeta$)")
ax2 = ax1.twinx()
ax2.plot(history["a"], label='Emergent Impedance (a)', color='red', alpha=0.3)
ax2.set_ylabel("Impedance 'a'")
plt.title("Dynamic Nucleon Resolution: The Smooth Snap")
plt.grid(True)
plt.show()