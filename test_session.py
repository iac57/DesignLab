import csv
import time
from NatNetClient import NatNetClient

# =============================================================================
# Parameters and Global Variables
# =============================================================================

CSV_FILE = "motion_data.csv"
DOWNSAMPLE_RATE = 10  # Log every 10th frame (e.g., 200Hz -> ~20Hz)

# Global buffer and state variables
csv_rows = []           # Buffer for all frames recorded while subject is in the foyer
frame_counter = 0       # Global counter for downsampled frames
session_state = "IN_FOYER"  # Possible states: IN_FOYER, TRANSITION, SESSION_COMPLETE
session_label = ""      # Slot machine label determined later

# Foyer boundary as a polygon (list of (x,y) tuples)
# Adjust these coordinates as needed
foyer_polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]

# Slot machine regions: each a tuple (label, polygon)
slot_machine_polygons = [
    ("SlotMachine1", [(12, 0), (15, 0), (15, 5), (12, 5)]),
    ("SlotMachine2", [(12, 6), (15, 6), (15, 11), (12, 11)])
]

# =============================================================================
# Helper Functions
# =============================================================================

def point_in_polygon(x, y, polygon):
    """
    Returns True if the point (x,y) is inside the polygon using the ray-casting algorithm.
    """
    num = len(polygon)
    j = num - 1
    inside = False
    for i in range(num):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def detect_slot_machine(position, slot_machine_polygons):
    """
    Returns the label for the first slot machine polygon in which the subject's (x,y) lies.
    If none match, returns "Unknown".
    """
    x, y = position[0], position[1]
    for label, poly in slot_machine_polygons:
        if point_in_polygon(x, y, poly):
            return label
    return "Unknown"

# =============================================================================
# Data Logging Callback with State Machine
# =============================================================================

def receive_rigid_body_frame(new_id, position, rotation):
    """
    Processes each rigid body frame. Depending on the current session_state:
    
      - IN_FOYER: Buffer frames (downsampled) if the subject is inside the foyer.
      - TRANSITION: No data is recorded; wait until a slot machine region is entered.
      - SESSION_COMPLETE: Optionally, a new session could be started if the subject re-enters the foyer.
    """
    global frame_counter, session_state, session_label, csv_rows

    frame_counter += 1
    if frame_counter % DOWNSAMPLE_RATE != 0:
        return  # Skip frames for downsampling

    # Check if subject is inside the foyer
    in_foyer = point_in_polygon(position[0], position[1], foyer_polygon)

    if session_state == "IN_FOYER":
        if in_foyer:
            # Buffer data: format numbers to 5 decimals
            formatted_position = [f"{p:.5f}" for p in position]
            formatted_rotation = [f"{r:.5f}" for r in rotation]
            # Buffer row: [frame, rigid body id, x, y, z, qw, qx, qy, qz, slot label (empty for now)]
            csv_rows.append([frame_counter, new_id] + formatted_position + formatted_rotation + [""])
        else:
            # Subject has just left the foyer. Switch to TRANSITION state.
            session_state = "TRANSITION"
            print("Subject left foyer. Waiting for slot machine region...")
    
    elif session_state == "TRANSITION":
        # Not recording data during transition.
        # Check if subject has now entered a slot machine region.
        detected_label = detect_slot_machine(position, slot_machine_polygons)
        if detected_label != "Unknown":
            session_label = detected_label
            # Update all buffered rows with the detected slot machine label.
            for row in csv_rows:
                row[-1] = session_label
            # Flush buffered data to CSV.
            with open(CSV_FILE, mode="a", newline="") as file:
                writer = csv.writer(file)
                for row in csv_rows:
                    writer.writerow(row)
            # Clear buffer and mark session as complete.
            csv_rows = []
            session_state = "SESSION_COMPLETE"
            print(f"Session complete: Slot machine '{session_label}' detected.")
    
    elif session_state == "SESSION_COMPLETE":
        # Optionally, you could check if the subject has returned to the foyer to start a new session.
        if in_foyer:
            session_state = "IN_FOYER"
            csv_rows = []  # Reset buffer for new session
            print("New session started: Subject re-entered the foyer.")

# =============================================================================
# CSV Initialization and Client Setup
# =============================================================================

# Write CSV header.
with open(CSV_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "RigidBodyID", "X", "Y", "Z", "QW", "QX", "QY", "QZ", "SlotMachine"])

# Setup NatNet client.
client = NatNetClient()
client.rigid_body_listener = receive_rigid_body_frame
client.run()

# =============================================================================
# Main Loop
# =============================================================================

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping data collection.")
    client.shutdown()
