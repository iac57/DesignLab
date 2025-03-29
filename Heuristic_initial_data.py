import pygame
import numpy as np
import csv
import os
import time

# Initialize pygame\pygame.init()

# Screen setup
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slot Machine Experiment")

# Define slot machine positions
M = 1 # sum of all machine probabilities
MACHINE_COUNT = 4
MACHINE_WIDTH = WIDTH // MACHINE_COUNT
machines = [(i * MACHINE_WIDTH, HEIGHT // 2, MACHINE_WIDTH, MACHINE_WIDTH) for i in range(MACHINE_COUNT)]

# Initial win probabilities (must sum to 1)
p_win = np.array([M/MACHINE_COUNT, M/MACHINE_COUNT, M/MACHINE_COUNT, M/MACHINE_COUNT])

# Data storage
data = []

# Experiment parameters
TOTAL_TRIALS = 30
alpha = 0.05  # Learning rate for probability updates

# Function to determine win/loss
def play_machine(machine_index):
    return np.random.rand() < p_win[machine_index]

# Run experiment
running = True
trial = 0
while running and trial < TOTAL_TRIALS:
    screen.fill((255, 255, 255))
    
    # Draw machines
    for i, rect in enumerate(machines):
        pygame.draw.rect(screen, (100, 100, 255), rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 3)  # Draw border
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            for i, rect in enumerate(machines):
                if rect[0] < x < rect[0] + rect[2]:
                    win = play_machine(i)
                    data.append((trial, i, win))
                    trial += 1
                    print(f"Trial {trial}: Machine {i} {'Win' if win else 'Loss'}")
                    
                    # Heuristic model: Adjust probabilities
                    if win:
                        p_win[i] = max(0.01, p_win[i] - alpha)  # Reduce win chance
                    else:
                        p_win[i] = min(0.99, p_win[i] + alpha)  # Increase win chance
                    
                    # Normalize to keep sum = 1
                    p_win /= p_win.sum()
                    p_win = M*p_win

# Save data to CSV
# Create a data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate a unique filename using timestamp
experiment_id = int(time.time())  # Unix timestamp for uniqueness
filename = f"data/slot_machine_data_{experiment_id}.csv"

# Save data with an experiment ID
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment_id", "trial", "machine", "win"])
    for row in data:
        writer.writerow([experiment_id, *row])  # Prepend experiment ID

print(f"Data saved to {filename}")

pygame.quit()
