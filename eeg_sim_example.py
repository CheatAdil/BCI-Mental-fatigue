import numpy as np
from eeg_simulator.eeg_sim import EEGSimulator
import time

sim = EEGSimulator()

print("Sample 1\n")
print(sim.get_next_sample())

time.sleep(2)
print("Sample 2\n")
print(sim.get_next_sample())

time.sleep(2)
print("Sample 3\n")
print(sim.get_next_sample())

