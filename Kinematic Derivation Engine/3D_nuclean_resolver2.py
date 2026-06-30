import math
import matplotlib.pyplot as plt

# --- 3D Nucleon Resolution Engine ---
# Logic: Pressure (a) is the arbiter of reality. 
# If pressure saturates the vacuum limit, the particle MUST yield via Tilt (zeta).

class Nucleon:
    def __init__(self, name, x_pos):
        self.name = name
        self.x = x_pos
        # Zeta: Coupling Constant (1.0 = Upright Proton, 0.0 = 90 deg Neutron)
        self.zeta = 1.0  
        # v_rad: Radial pulse (Charge)
        self.v_rad = 1.0 
        self.is_neutron = False

    def update(self, repulsion_force, yield_limit):
        # 1. Calculate Local Impedance (a)
        # Higher repulsion = higher pressure/impedance
        a = abs(repulsion_force) 
        
        # 2. Check for Saturation (The Hard Limit)
        if a > yield_limit:
            # Saturation detected: The particle must yield
            # Apply torque to zeta (Tilt towards 90 deg / 0.0 coupling)
            if self.zeta > 0.1:
                self.zeta -= 0.05  # Roll towards neutron state
                self.v_rad = 0.0   # Pulse is masked by tilt
                self.is_neutron = True
            else:
                self.zeta = 0.0
                self.is_neutron = True

# --- Simulation Setup ---
p1 = Nucleon("Proton_1", 0)
p2 = Nucleon("Proton_2", 1.0) # They start very close

vacuum_yield_limit = 0.5
history = {"p2_zeta": [], "p2_vrad": [], "repulsion": []}

print("Initiating Nucleon Snap...")

# Run Loop
for t in range(100):
    # Calculate repulsive pulse clash
    repulsion = (p1.v_rad * p2.v_rad) / (abs(p1.x - p2.x) + 0.1)
    
    # Update state based on saturation
    p2.update(repulsion, vacuum_yield_limit)
    
    # Track
    history["p2_zeta"].append(p2.zeta)
    history["p2_vrad"].append(p2.v_rad)
    history["repulsion"].append(repulsion)

print(f"Final State: P2 is_neutron = {p2.is_neutron}, Final Zeta = {p2.zeta:.2f}")

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(history["p2_zeta"], color='blue', label='Zeta (Tilt State)')
ax1.set_title("Nucleon Resolution: The Snap")
ax1.set_ylabel("Coupling Constant ($\zeta$)")
ax1.grid(True)

ax2.plot(history["repulsion"], color='red', label='Repulsive Pulse')
ax2.plot(history["p2_vrad"], color='purple', label='Radial Pulse ($v_{rad}$)')
ax2.set_xlabel("Steps")
ax2.set_ylabel("Intensity")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()