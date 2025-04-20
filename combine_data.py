import pandas as pd
import glob

# Load all experiment data
file_list = glob.glob("combined_data/slot_machine_data_*.csv")
all_data = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

# Save combined data for training
all_data.to_csv("slot_machine_data.csv", index=False)

print("All experiment data combined and saved.")