import csv
import time
from NatNetClient import NatNetClient
import MoCapData  # This provides the RigidBodyData class definition

# File to write downsampled rigid body data
CSV_FILE = "downsampled_rigid_body_data.csv"

# We want to downsample: record 1 frame every 200 frames (since base frequency is 200 Hz)
FRAME_DOWNSAMPLE = 200

# Global frame counter
frame_counter = 0

def new_frame_callback(data_dict):
    """
    Callback called once per full frame.
    The data_dict is expected to contain a key "rigid_body_data"
    that holds a MoCapData.RigidBodyData instance.
    """
    global frame_counter
    frame_counter += 1

    # Only process every FRAME_DOWNSAMPLE-th frame
    if frame_counter % FRAME_DOWNSAMPLE != 0:
        return

    # Get the rigid body data from the current frame.
    rigid_body_data = data_dict.get("rigid_body_data", None)
    print_rigid_body_data(rigid_body_data, frame_counter)

def print_rigid_body_data(rigid_body_data, frame):
    """
    Mimics the get_as_string method of RigidBodyData.
    For each rigid body in the frame, print its ID, position, and rotation
    (all formatted to 5 decimal places) and write the same info to a CSV file.
    """
    if rigid_body_data is None:
        print(f"Frame {frame}: No rigid body data.")
        return

    # Build output string
    output = f"Frame {frame} - Rigid Body Data:\n"
    csv_rows = []
    # Iterate over each rigid body in the rigid_body_data list.
    for rb in rigid_body_data.rigid_body_list:
        # rb is an instance of MoCapData.RigidBody, which has attributes: id_num, pos, rot
        pos = rb.pos
        rot = rb.rot
        output += (f"  Rigid Body {rb.id_num}: Position: "
                   f"({pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}), Rotation: "
                   f"({rot[0]:.5f}, {rot[1]:.5f}, {rot[2]:.5f}, {rot[3]:.5f})\n")
        # Prepare a CSV row for this rigid body.
        csv_rows.append([frame, rb.id_num,
                         f"{pos[0]:.5f}", f"{pos[1]:.5f}", f"{pos[2]:.5f}",
                         f"{rot[0]:.5f}", f"{rot[1]:.5f}", f"{rot[2]:.5f}", f"{rot[3]:.5f}"])
    # Print the formatted string to the screen.
    print(output)

    # Append the rows to the CSV file.
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        for row in csv_rows:
            writer.writerow(row)

if __name__ == "__main__":
    # Write CSV header.
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "RigidBodyID", "X", "Y", "Z", "QW", "QX", "QY", "QZ"])

    # Create the NatNet client and assign our new frame listener.
    client = NatNetClient()
    client.new_frame_listener = new_frame_callback

    # Start the streaming client.
    client.run()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping data collection.")
        client.shutdown()
