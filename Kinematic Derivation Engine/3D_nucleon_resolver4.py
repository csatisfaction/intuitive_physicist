import numpy as np
import matplotlib.pyplot as plt

class Nucleon:
    def __init__(self, x_pos, velocity):
        self.x = x_pos
        self.v = velocity
        self.zeta = 1.0
        self.radius = 0.2 # Adjusted for tighter collision
        
    def get_forces(self, other):
        dist = abs(self.x - other.x)
        contact = self.radius * 2
        # Emergent pressure: Repulsive if overlapping, attractive if bridging
        if dist < contact:
            # Stiff repulsion
            a = 0.5 / (dist - (contact * 0.9))**2
        else:
            # Weak attraction
            a = -0.05 / (dist**2)
        return a

# Setup
p1 = Nucleon(0, 0)
p2 = Nucleon(0.45, -0.01) # p2 is moving toward p1

history = {"zeta": [], "dist": [], "a": []}

for t in range(200):
    # 1. Physics: Interaction
    force = p2.get_forces(p1)
    
    # 2. Integration: Update position based on force
    p2.v += force * 0.001
    p2.x += p2.v
    
    # 3. Evolution: Zeta responds to local pressure (torque)
    torque = abs(force) * 0.1 # Increased sensitivity
    p2.zeta = np.clip(p2.zeta - torque, 0.0, 1.0)
    
    # Track
    history["zeta"].append(p2.zeta)
    history["dist"].append(abs(p2.x - p1.x))
    history["a"].append(force)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(history["zeta"], label='Zeta (Tilt)', color='blue')
ax.plot(np.array(history["a"])*10, label='Impedance (a) * 10', color='red', alpha=0.3)
ax.set_title("The Snap: Emergent Resolution")
ax.legend()
plt.grid(True)
plt.show()