import pygame
import numpy as np
import csv
import os
import time

# Initialize pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slot Machine Experiment")

# Define slot machine positions
M = 2  # Sum of all machine probabilities
MACHINE_COUNT = 4
MACHINE_WIDTH = WIDTH // MACHINE_COUNT
machines = [(i * MACHINE_WIDTH, HEIGHT // 2, MACHINE_WIDTH, MACHINE_WIDTH) for i in range(MACHINE_COUNT)]

# Reset button position
RESET_BUTTON_RECT = pygame.Rect(WIDTH // 2 - 50, HEIGHT - 50, 100, 30)

# Initial win probabilities (must sum to 1)
p_win = np.array([M / MACHINE_COUNT] * MACHINE_COUNT)

# Data storage
data = []

# Experiment parameters
TOTAL_TRIALS = 30
alpha = 0.05  # Learning rate for probability updates

# Set up font for displaying text
font = pygame.font.SysFont('Arial', 24)

# Function to determine win/loss
def play_machine(machine_index):
    return np.random.rand() < p_win[machine_index]

# Run experiment
running = True
trial = 0
last_win = None  # Store last win/loss outcome
waiting_for_reset = False  # Controls whether reset is needed
countdown_start = None  # Tracks when countdown begins
countdown_time = 2  # 2-second countdown

while running and trial < TOTAL_TRIALS:
    screen.fill((255, 255, 255))
    
    # Draw slot machines
    for i, rect in enumerate(machines):
        pygame.draw.rect(screen, (100, 100, 255), rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 3)  # Border

    # Display last trial result
    if last_win is not None:
        trial_text = font.render(f"Trial {trial}: {last_win}", True, (0, 0, 0))
        screen.blit(trial_text, (WIDTH // 2 - trial_text.get_width() // 2, HEIGHT // 4))
    
    # Draw reset button
    pygame.draw.rect(screen, (200, 0, 0), RESET_BUTTON_RECT)
    reset_text = font.render("Reset", True, (255, 255, 255))
    screen.blit(reset_text, (RESET_BUTTON_RECT.x + 20, RESET_BUTTON_RECT.y + 5))

    # Handle countdown
    if countdown_start is not None:
        elapsed = time.time() - countdown_start
        if elapsed < countdown_time:
            countdown_text = font.render(f"Next trial in {round(countdown_time - elapsed)}...", True, (0, 0, 0))
            screen.blit(countdown_text, (WIDTH // 2 - 60, HEIGHT // 3))
        else:
            countdown_start = None  # Countdown finished, allow next trial

    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            # Reset button click (only when waiting for reset)
            if waiting_for_reset and RESET_BUTTON_RECT.collidepoint(x, y):
                waiting_for_reset = False
                countdown_start = time.time()  # Start countdown
                print("Reset clicked. Countdown started.")

            # Machine click (only when countdown is over and reset is not needed)
            elif not waiting_for_reset and countdown_start is None:
                machine_clicked = False
                for i, rect in enumerate(machines):
                    if rect[0] < x < rect[0] + rect[2] and rect[1] < y < rect[1] + rect[3]:
                        win = play_machine(i)
                        data.append((trial, i, win))
                        trial += 1
                        last_win = "Win" if win else "Loss"
                        print(f"Trial {trial}: Machine {i} {'Win' if win else 'Loss'}")

                        # Adjust probabilities
                        if win:
                            p_win[i] = max(0.01, p_win[i] - alpha)  # Reduce win chance
                        else:
                            p_win[i] = min(0.99, p_win[i] + alpha)  # Increase win chance
                        
                        # Normalize probabilities
                        p_win /= p_win.sum()
                        p_win = M * p_win

                        waiting_for_reset = True  # Require reset before next trial
                        machine_clicked = True
                        break  # Only one machine can be clicked per trial

                # Ignore clicks outside valid areas (machines or reset button)
                if not machine_clicked:
                    print("Invalid click: Not on a machine or reset button.")
                    
# Display total wins before quitting
total_wins = sum(1 for _, _, win in data if win)
screen.fill((255, 255, 255))
final_text = font.render(f"Experiment complete! Total wins: {total_wins}/{TOTAL_TRIALS}", True, (0, 0, 0))
screen.blit(final_text, (WIDTH // 2 - final_text.get_width() // 2, HEIGHT // 2))
pygame.display.flip()

# Pause for a few seconds to let the player see the result
time.sleep(5)

# Save data to CSV
os.makedirs("data", exist_ok=True)
experiment_id = int(time.time())  # Unique ID
filename = f"data/slot_machine_data_{experiment_id}.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["experiment_id", "trial", "machine", "win"])
    for row in data:
        writer.writerow([experiment_id, *row])  # Prepend experiment ID

print(f"Data saved to {filename}")

pygame.quit()
