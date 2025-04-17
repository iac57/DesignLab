#Copyright © 2018 Naturalpoint
#
#Licensed under the Apache License, Version 2.0 (the "License")
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.
import csv
import socket
import sys
import time
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData
from PlayMachine import PlayMachine, machines
from shapely.geometry import Point
from FoyerDetector import FoyerDetector
import os
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
#from model_loader import get_predictor
# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.
def receive_new_frame(data_dict):
    order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
    dump_args = False
    if dump_args == True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict :
                out_string += data_dict[key] + " "
            out_string+="/"
        print(out_string)

#predictor = get_predictor()
csv_playlog = "machine_play_log.csv"
trial_file = "rigid_body_data.csv"
last_machine_id = 0
last_play_time = 0  # Stores last play timestamp in seconds
FRAME_COUNTER = 0  # Single counter for both rigid bodies
TOTAL_FRAMES = 0
was_outof_foyer = False #Flag to check if the body is in the foyer
trial_number = 1 #Initialize trial number
torso_rigid_body_id = 1  # ID for torso rigid body
head_rigid_body_id = 2   # ID for head rigid body
current_frame_data = {}  # Store data for current frame
last_foyer_state = None  # Track last foyer state to prevent duplicate messages
# Sampling rate configuration
MOCAP_FRAMERATE = 200  # Motive's capture rate in Hz
SAMPLING_RATE = 10  # (Hz)
FRAME_INTERVAL = MOCAP_FRAMERATE // SAMPLING_RATE  # Automatically calculate frames to skip

# To change sampling rate, just modify SAMPLING_RATE
# For example:
# SAMPLING_RATE = 10  # For 10 samples per second
# SAMPLING_RATE = 20  # For 20 samples per second

#Another callback method. This function is called once per rigid body per frame
def classify_torso_angle(quaternion):
    euler_angles = R.from_quat(quaternion).as_euler('zyx', degrees=True)
    torso_angle = euler_angles[1]
    if torso_angle < 0:
        answer = 12
    else:
        answer = 34
    return answer

def receive_rigid_body_frame(new_id, position, rotation):
    global last_machine_id, last_play_time, FRAME_COUNTER, trial_number, was_outof_foyer, torso_rigid_body_id, head_rigid_body_id, TOTAL_FRAMES, FRAME_INTERVAL, current_frame_data, last_foyer_state
    
    # Store data for this rigid body in the current frame
    current_frame_data[new_id] = {
        'position': position,
        'rotation': rotation
    }
    
    # If we have both rigid bodies for this frame, process them
    if len(current_frame_data) == 2:  # We have both torso and head data
        TOTAL_FRAMES += 1
        FRAME_COUNTER += 1
        
        # Write debug info to log file
        with open("debug_log.txt", "a") as log_file:
            if new_id not in current_frame_data:
                log_file.write(f"Frame {TOTAL_FRAMES}: Received data for rigid body ID: {new_id}\n")
            log_file.write(f"Frame {TOTAL_FRAMES}: Processing frame with rigid bodies: {list(current_frame_data.keys())}\n")
        
        # Process torso data
        if torso_rigid_body_id in current_frame_data:
            torso_data = current_frame_data[torso_rigid_body_id]
            body_cm = Point(torso_data['position'][0], torso_data['position'][2])
            
            loc = FoyerDetector()
            prediction = None
            
            # Check current foyer state
            current_foyer_state = loc.is_in_foyer(body_cm)
            
            # Log state changes
            if current_foyer_state != last_foyer_state: #If player enters/leaves foyer
                with open("debug_log.txt", "a") as log_file, open("prediction_log.txt", "a") as prediction_file:
                    if current_foyer_state: #If player is in foyer
                        log_file.write(f"Frame {TOTAL_FRAMES}: Body has entered the foyer\n")
                        # Only increment trial if we were previously out of the foyer
                        if was_outof_foyer: #This means a new trial has started
                            trial_number += 1
                            FRAME_COUNTER = 0 #!! Reset sampling logic for next trial 
                            log_file.write(f"Frame {TOTAL_FRAMES}: Trial number incremented to {trial_number}\n")
                            was_outof_foyer = False
                            last_machine_id = 0  # Reset machine ID for new trial
                            log_file.write(f"Frame {TOTAL_FRAMES}: Body has reentered the foyer\n")
                    
                    #Make MoCap rediction as soon as player leaves the foyer"
                    else: #If player is out of the foyer
                        data_df = pd.read_csv(trial_file) #since rigid_body_data is being continuously updated, i need to continuously read it into a dataframe
                        #result = predictor.process_frame(data_df) #this should work as it will just take the last n samples as a feature vector
                       # prediction_file.write(f"Prediction for Trial {trial_number}: {result}\n")
                        log_file.write(f"Frame {TOTAL_FRAMES}: Body has left the foyer\n")
                        log_file.write(f"Frame {TOTAL_FRAMES}: Checking for machines...\n")
                        was_outof_foyer = True  # Set this when body leaves foyer
                last_foyer_state = current_foyer_state
            
            if current_foyer_state:
                if FRAME_COUNTER == FRAME_INTERVAL: #this only happens once unless FRAME_COUNTER gets reset to 0 for each trial
                    FRAME_COUNTER = 0
                    filename = "rigid_body_data.csv"
                    #prediction = classify_torso_angle(torso_data['rotation'])
                    file_exists = os.path.isfile(filename)
                    
                    with open(filename, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        if not file_exists:
                            writer.writerow(["Trial Number", "Frame Number", "Rigid Body ID", "Position X", "Position Y", "Position Z",
                                        "Orientation W", "Orientation X", "Orientation Y", "Orientation Z"])
                        # Write head data first
                        if head_rigid_body_id in current_frame_data:
                            head_data = current_frame_data[head_rigid_body_id]
                            writer.writerow([trial_number, TOTAL_FRAMES, head_rigid_body_id, *head_data['position'], *head_data['rotation']])
                            with open("debug_log.txt", "a") as log_file:
                                log_file.write(f"Frame {TOTAL_FRAMES}: Writing head data (ID {head_rigid_body_id})\n")
                        else:
                            with open("debug_log.txt", "a") as log_file:
                                log_file.write(f"Frame {TOTAL_FRAMES}: Warning: Head data (ID {head_rigid_body_id}) not found in current frame\n")
                        # Write torso data second
                        writer.writerow([trial_number, TOTAL_FRAMES, torso_rigid_body_id, *torso_data['position'], *torso_data['rotation']])
                        with open("debug_log.txt", "a") as log_file:
                            log_file.write(f"Frame {TOTAL_FRAMES}: Writing torso data (ID {torso_rigid_body_id})\n")
            else:
                # Only check for machine play if we're not in the foyer
                machine_id, won = PlayMachine(machines, body_cm)
                if machine_id > 0 and last_machine_id == 0:  # Only process first machine play
                    current_time = time.time()
                    if current_time - last_play_time >= 5:  # 5 second cooldown
                        with open("debug_log.txt", "a") as log_file:
                            log_file.write(f"Frame {TOTAL_FRAMES}: MACHINE {machine_id} PLAYED in trial {trial_number}\n")
                            if won:
                                log_file.write(f"Frame {TOTAL_FRAMES}: You won!\n")
                            else:
                                log_file.write(f"Frame {TOTAL_FRAMES}: Loser\n")
                        file_exists = os.path.isfile(csv_playlog)
                        with open(csv_playlog, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            if not file_exists:
                                writer.writerow(["Trial Number", "Frame", "Machine ID", "Reward", "Prediction"])
                            # Use the same frame number as the last written rigid body data
                            last_written_frame = TOTAL_FRAMES - (TOTAL_FRAMES % FRAME_INTERVAL)
                            writer.writerow([trial_number, last_written_frame, machine_id, won, prediction])
                        last_play_time = current_time
                        last_machine_id = machine_id  # Mark that we've played a machine this trial
        
        # Clear the current frame data for the next frame
        current_frame_data = {}

def add_lists(totals, totals_tmp):
    totals[0]+=totals_tmp[0]
    totals[1]+=totals_tmp[1]
    totals[2]+=totals_tmp[2]
    return totals

def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:          %s"% natnet_client.local_ip_address)
    print("  Server:          %s"% natnet_client.server_ip_address)
    print("  Command Port:    %d"% natnet_client.command_port)
    print("  Data Port:       %d"% natnet_client.data_port)

    changeBitstreamString = "  Can Change Bitstream Version = "
    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s"% natnet_client.multicast_address)
        changeBitstreamString+="false"
    else:
        print("  Using Unicast")
        changeBitstreamString+="true"

    #NatNet Server Info
    application_name = natnet_client.get_application_name()
    nat_net_requested_version = natnet_client.get_nat_net_requested_version()
    nat_net_version_server = natnet_client.get_nat_net_version_server()
    server_version = natnet_client.get_server_version()

    print("  NatNet Server Info")
    print("    Application Name %s" %(application_name))
    print("    MotiveVersion  %d %d %d %d"% (server_version[0], server_version[1], server_version[2], server_version[3]))
    print("    NatNetVersion  %d %d %d %d"% (nat_net_version_server[0], nat_net_version_server[1], nat_net_version_server[2], nat_net_version_server[3]))
    print("  NatNet Bitstream Requested")
    print("    NatNetVersion  %d %d %d %d"% (nat_net_requested_version[0], nat_net_requested_version[1],\
       nat_net_requested_version[2], nat_net_requested_version[3]))

    print(changeBitstreamString)
    #print("command_socket = %s"%(str(natnet_client.command_socket)))
    #print("data_socket    = %s"%(str(natnet_client.data_socket)))
    print("  PythonVersion    %s"%(sys.version))


def print_commands(can_change_bitstream):
    outstring = "Commands:\n"
    outstring += "Return Data from Motive\n"
    outstring += "  s  send data descriptions\n"
    outstring += "  r  resume/start frame playback\n"
    outstring += "  p  pause frame playback\n"
    outstring += "     pause may require several seconds\n"
    outstring += "     depending on the frame data size\n"
    outstring += "Change Working Range\n"
    outstring += "  o  reset Working Range to: start/current/end frame 0/0/end of take\n"
    outstring += "  w  set Working Range to: start/current/end frame 1/100/1500\n"
    outstring += "Return Data Display Modes\n"
    outstring += "  j  print_level = 0 supress data description and mocap frame data\n"
    outstring += "  k  print_level = 1 show data description and mocap frame data\n"
    outstring += "  l  print_level = 20 show data description and every 20th mocap frame data\n"
    outstring += "Change NatNet data stream version (Unicast only)\n"
    outstring += "  3  Request NatNet 3.1 data stream (Unicast only)\n"
    outstring += "  4  Request NatNet 4.1 data stream (Unicast only)\n"
    outstring += "General\n"
    outstring += "  t  data structures self test (no motive/server interaction)\n"
    outstring += "  c  print configuration\n"
    outstring += "  h  print commands\n"
    outstring += "  q  quit\n"
    outstring += "\n"
    outstring += "NOTE: Motive frame playback will respond differently in\n"
    outstring += "       Endpoint, Loop, and Bounce playback modes.\n"
    outstring += "\n"
    outstring += "EXAMPLE: PacketClient [serverIP [ clientIP [ Multicast/Unicast]]]\n"
    outstring += "         PacketClient \"192.168.10.14\" \"192.168.10.14\" Multicast\n"
    outstring += "         PacketClient \"127.0.0.1\" \"127.0.0.1\" u\n"
    outstring += "\n"
    print(outstring)

def request_data_descriptions(s_client):
    # Request the model definitions
    s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF,    "",  (s_client.server_ip_address, s_client.command_port) )

def test_classes():
    totals = [0,0,0]
    print("Test Data Description Classes")
    totals_tmp = DataDescriptions.test_all()
    totals=add_lists(totals, totals_tmp)
    print("")
    print("Test MoCap Frame Classes")
    totals_tmp = MoCapData.test_all()
    totals=add_lists(totals, totals_tmp)
    print("")
    print("All Tests totals")
    print("--------------------")
    print("[PASS] Count = %3.1d"%totals[0])
    print("[FAIL] Count = %3.1d"%totals[1])
    print("[SKIP] Count = %3.1d"%totals[2])

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict


if __name__ == "__main__":

    optionsDict = {}
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    optionsDict["clientAddress"] = IPAddr
    optionsDict["serverAddress"] = "10.229.139.24"
    optionsDict["use_multicast"] = True

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])
    streaming_client.set_print_level(0)  # Suppress frame printing

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    #Configure the streaming client to call receive_rigid_body_frame method on each rigid body in a frame. See _unpack_rigid_body() method in NatNetClient.py
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    is_looping = True
    time.sleep(1)
    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    print_configuration(streaming_client)
    print("\n")
    print_commands(streaming_client.can_change_bitstream_version())


    while is_looping:
        inchars = input('Enter command or (\'h\' for list of commands)\n')
        if len(inchars)>0:
            c1 = inchars[0].lower()
            if c1 == 'h' :
                print_commands(streaming_client.can_change_bitstream_version())
            elif c1 == 'c' :
                print_configuration(streaming_client)
            elif c1 == 's':
                request_data_descriptions(streaming_client)
                time.sleep(1)
            elif (c1 == '3') or (c1 == '4'):
                if streaming_client.can_change_bitstream_version():
                    tmp_major = 4
                    tmp_minor = 1
                    if(c1 == '3'):
                        tmp_major = 3
                        tmp_minor = 1
                    return_code = streaming_client.set_nat_net_version(tmp_major,tmp_minor)
                    time.sleep(1)
                    if return_code == -1:
                        print("Could not change bitstream version to %d.%d"%(tmp_major,tmp_minor))
                    else:
                        print("Bitstream version at %d.%d"%(tmp_major,tmp_minor))
                else:
                    print("Can only change bitstream in Unicast Mode")

            elif c1 == 'p':
                sz_command="TimelineStop"
                return_code = streaming_client.send_command(sz_command)
                time.sleep(1)
                print("Command: %s - return_code: %d"% (sz_command, return_code) )
            elif c1 == 'r':
                sz_command="TimelinePlay"
                return_code = streaming_client.send_command(sz_command)
                print("Command: %s - return_code: %d"% (sz_command, return_code) )
            elif c1 == 'o':
                tmpCommands=["TimelinePlay",
                            "TimelineStop",
                            "SetPlaybackStartFrame,0",
                            "SetPlaybackStopFrame,1000000",
                            "SetPlaybackLooping,0",
                            "SetPlaybackCurrentFrame,0",
                            "TimelineStop"]
                for sz_command in tmpCommands:
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d"% (sz_command, return_code) )
                time.sleep(1)
            elif c1 == 'w':
                tmp_commands=["TimelinePlay",
                            "TimelineStop",
                            "SetPlaybackStartFrame,1",
                            "SetPlaybackStopFrame,1500",
                            "SetPlaybackLooping,0",
                            "SetPlaybackCurrentFrame,100",
                            "TimelineStop"]
                for sz_command in tmp_commands:
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d"% (sz_command, return_code) )
                time.sleep(1)
            elif c1 == 't':
                test_classes()

            elif c1 == 'j':
                streaming_client.set_print_level(0)
                print("Showing only received frame numbers and supressing data descriptions")
            elif c1 == 'k':
                streaming_client.set_print_level(1)
                print("Showing every received frame")

            elif c1 == 'l':
                print_level = streaming_client.set_print_level(20)
                print_level_mod = print_level % 100
                if(print_level == 0):
                    print("Showing only received frame numbers and supressing data descriptions")
                elif (print_level == 1):
                    print("Showing every frame")
                elif (print_level_mod == 1):
                    print("Showing every %dst frame"%print_level)
                elif (print_level_mod == 2):
                    print("Showing every %dnd frame"%print_level)
                elif (print_level == 3):
                    print("Showing every %drd frame"%print_level)
                else:
                    print("Showing every %dth frame"%print_level)

            elif c1 == 'q':
                is_looping = False
                streaming_client.shutdown()
                break
            else:
                print("Error: Command %s not recognized"%c1)
            print("Ready...\n")
    print("exiting")
