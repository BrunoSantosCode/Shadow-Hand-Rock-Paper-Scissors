#!/usr/bin/env python3

import rospy
from termcolor import colored
from sr_robot_commander.sr_hand_commander import SrHandCommander

# Shadow Hand Poses

# Rock
rock_hand_prev = {'rh_FFJ1': 1.571, 'rh_FFJ2': 1.571, 'rh_FFJ3': 1.571, 'rh_FFJ4': 0.000,
                  'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
                  'rh_MFJ1': 1.571, 'rh_MFJ2': 1.571, 'rh_MFJ3': 1.571, 'rh_MFJ4': 0.000,
                  'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
                  'rh_THJ1': 0.300, 'rh_THJ2': 0.250, 'rh_THJ3': 0.000, 'rh_THJ4': 1.221, 'rh_THJ5': 0.050,
                  'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

rock_hand = {'rh_FFJ1': 1.571, 'rh_FFJ2': 1.571, 'rh_FFJ3': 1.571, 'rh_FFJ4': 0.000,
             'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
             'rh_MFJ1': 1.571, 'rh_MFJ2': 1.571, 'rh_MFJ3': 1.571, 'rh_MFJ4': 0.000,
             'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
             'rh_THJ1': 0.730, 'rh_THJ2': 0.698, 'rh_THJ3': 0.000, 'rh_THJ4': 1.221, 'rh_THJ5': 0.050,
             'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Paper
paper_hand = {'rh_FFJ1': 0.000, 'rh_FFJ2': 0.000, 'rh_FFJ3': 0.000, 'rh_FFJ4': 0.000, 
              'rh_LFJ1': 0.000, 'rh_LFJ2': 0.000, 'rh_LFJ3': 0.000, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
              'rh_MFJ1': 0.000, 'rh_MFJ2': 0.000, 'rh_MFJ3': 0.000, 'rh_MFJ4': 0.000,
              'rh_RFJ1': 0.000, 'rh_RFJ2': 0.000, 'rh_RFJ3': 0.000, 'rh_RFJ4': 0.000,
              'rh_THJ1': 0.000, 'rh_THJ2': 0.000, 'rh_THJ3': 0.000, 'rh_THJ4': 0.000, 'rh_THJ5': 0.000,
              'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Scissor
scissors_hand = {'rh_FFJ1': 0.000, 'rh_FFJ2': 0.000, 'rh_FFJ3': 0.000, 'rh_FFJ4': -0.180,
                'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
                'rh_MFJ1': 0.000, 'rh_MFJ2': 0.000, 'rh_MFJ3': 0.000, 'rh_MFJ4': 0.180,
                'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
                'rh_THJ1': 0.730, 'rh_THJ2': 0.698, 'rh_THJ3': 0.000, 'rh_THJ4': 1.221, 'rh_THJ5': 0.050,
                'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Wrist Up
wrist_up = {'rh_WRJ1': 0.000, 'rh_WRJ2': 0.174}

# Wrist Down
wrist_down = {'rh_WRJ1': 0.000, 'rh_WRJ2': -0.523}


def set_hand_pose(pose: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand
        @param pose - A given object 'rock', 'paper' or 'scissors'
    """
    global hand_commander

    if pose == 'rock':
        hand_joints = hand_commander.get_joints_position()
        sum_ff = (hand_joints['rh_FFJ2']+hand_joints['rh_FFJ3'])*180/3.1415
        sum_mf = (hand_joints['rh_MFJ2']+hand_joints['rh_MFJ3'])*180/3.1415
        if ((sum_ff < 120) or (sum_mf < 120)): 
            hand_commander.move_to_joint_value_target_unsafe(joint_states=rock_hand_prev, time=1.0, wait=wait, angle_degrees=False)
        hand_commander.move_to_joint_value_target_unsafe(joint_states=rock_hand, time=1.0, wait=wait, angle_degrees=False)
        print('\n' + colored('Rock!', 'green') + '\n')
    elif pose == 'paper':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=paper_hand, time=1.0, wait=wait, angle_degrees=False)
        print('\n' + colored('Paper!', 'green') + '\n')
    elif pose == 'scissors':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=scissors_hand, time=1.0, wait=wait, angle_degrees=False)
        print('\n' + colored('Scissors!', 'green') + '\n')
    else:
        print('\n' + colored('ERROR: "' + pose + '" hand pose is not defined!', 'red') + '\n')


def set_wrist_pose(pose: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand wrist
        @param pose - A given pose 'up' or 'down'
    """
    global hand_commander

    if pose == 'up':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_up, time=1.0, wait=wait, angle_degrees=False)
        print('\n' + colored('Rock!', 'green') + '\n') 
    elif pose == 'down':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_down, time=1.0, wait=wait, angle_degrees=False)
        print('\n' + colored('Paper!', 'green') + '\n') 
    else:
        print('\n' + colored('ERROR: "' + pose + '" wrist pose is not defined!', 'red') + '\n')

def give_me_a(object: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand wrist
        @param object - A given object 'rock', 'paper' or 'scissors'.
    """
    set_hand_pose('rock', wait=wait)
    set_wrist_pose('up', wait=wait)
    set_wrist_pose('down', wait=wait)
    set_wrist_pose('up', wait=wait)
    set_wrist_pose('down', wait=wait)
    set_wrist_pose('up', wait=wait)
    set_hand_pose(pose = object, wait=wait)



if __name__ == "__main__":
    global hand_commander


    # Init ROS
    rospy.init_node('rock_paper_scissors')

    # Shadow Hand commander
    hand_commander = SrHandCommander(name='right_hand')

    # Set control velocity and acceleration
    hand_commander.set_max_velocity_scaling_factor(1.0)
    hand_commander.set_max_acceleration_scaling_factor(1.0)

    print('\n' + colored('"rock_paper_scissors" ROS node is ready!', 'green') + '\n') 

    while not rospy.is_shutdown():

        give_me_a('rock')
        rospy.sleep(2.0)
        give_me_a('paper')
        rospy.sleep(2.0)
        give_me_a('scissors')
        rospy.sleep(2.0)
