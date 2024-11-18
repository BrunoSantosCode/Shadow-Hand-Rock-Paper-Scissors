#!/usr/bin/env python3

import rospy
import random
from termcolor import colored
from sr_robot_commander.sr_hand_commander import SrHandCommander

# Shadow Hand Poses

# Default

default_hand = {'rh_FFJ1': 1.571, 'rh_FFJ2': 1.571, 'rh_FFJ3': 1.571, 'rh_FFJ4': 0.000,
                'rh_LFJ1': 1.571, 'rh_LFJ2': 1.571, 'rh_LFJ3': 1.571, 'rh_LFJ4': 0.000, 'rh_LFJ5': 0.000,
                'rh_MFJ1': 1.571, 'rh_MFJ2': 1.571, 'rh_MFJ3': 1.571, 'rh_MFJ4': 0.000,
                'rh_RFJ1': 1.571, 'rh_RFJ2': 1.571, 'rh_RFJ3': 1.571, 'rh_RFJ4': 0.000,
                'rh_THJ1': 0.716, 'rh_THJ2': 0.691, 'rh_THJ3': 0.201, 'rh_THJ4': 1.226, 'rh_THJ5': 0.186,
                'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

hand_aux = {'rh_THJ1': 0.250, 'rh_THJ2': 0.250}

# Rock

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

# Default Wrist
wrist_default = {'rh_WRJ1': 0.000, 'rh_WRJ2': 0.000}

# Wrist Up
wrist_up = {'rh_WRJ1': 0.000, 'rh_WRJ2': 0.174}

# Wrist Down
wrist_down = {'rh_WRJ1': 0.000, 'rh_WRJ2': -0.523}


def set_hand_pose(pose: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand
        @param pose - A given object 'default', 'rock', 'paper' or 'scissors'
    """
    global hand_commander

    if (pose == 'default') or (pose == 'rock'):
        hand_joints = hand_commander.get_joints_position()
        sum_ff = (hand_joints['rh_FFJ2']+hand_joints['rh_FFJ3'])*180/3.1415
        sum_mf = (hand_joints['rh_MFJ2']+hand_joints['rh_MFJ3'])*180/3.1415
        if ((sum_ff < 120) or (sum_mf < 120)): 
            hand_commander.move_to_joint_value_target_unsafe(joint_states=hand_aux, time=1.0, wait=wait, angle_degrees=False)
        if pose == 'default':
            hand_commander.move_to_joint_value_target_unsafe(joint_states=default_hand, time=1.0, wait=wait, angle_degrees=False)
        elif pose == 'rock':
            hand_commander.move_to_joint_value_target_unsafe(joint_states=rock_hand, time=1.0, wait=wait, angle_degrees=False)
    elif pose == 'paper':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=paper_hand, time=1.0, wait=wait, angle_degrees=False)
    elif pose == 'scissors':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=hand_aux, time=1.0, wait=wait, angle_degrees=False)
        hand_commander.move_to_joint_value_target_unsafe(joint_states=scissors_hand, time=1.0, wait=wait, angle_degrees=False)
    else:
        print('\n' + colored('ERROR: "' + pose + '" hand pose is not defined!', 'red') + '\n')


def set_wrist_pose(pose: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand wrist
        @param pose - A given pose 'up' or 'down'
    """
    global hand_commander

    if pose == 'default':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_default, time=1.0, wait=wait, angle_degrees=False)
    elif pose == 'up':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_up, time=1.0, wait=wait, angle_degrees=False)
    elif pose == 'down':
        hand_commander.move_to_joint_value_target_unsafe(joint_states=wrist_down, time=1.0, wait=wait, angle_degrees=False)
    else:
        print('\n' + colored('ERROR: "' + pose + '" wrist pose is not defined!', 'red') + '\n')

def give_me_a(object: str, wait = True):
    """
        Sets a pre-defined pose for Shadow Hand wrist
        @param object - A given object 'rock', 'paper' or 'scissors'.
    """
    set_hand_pose('default', wait=wait)
    set_wrist_pose('up', wait=wait)
    set_wrist_pose('down', wait=wait)
    set_wrist_pose('up', wait=wait)
    set_wrist_pose('down', wait=wait)
    set_wrist_pose('up', wait=wait)
    set_wrist_pose('default', wait=wait)
    set_hand_pose(pose = object, wait=wait)


def game_round():
    """
        Plays one round of the game.
        The robot picks a random move, the user enters their move, 
        and the result is displayed.
    """

    # Robot randomly selects a move
    moves = ['rock', 'paper', 'scissors']
    hand_move = random.choice(moves)
    give_me_a(hand_move)
    print('\n' + colored('Robot move: '+hand_move, 'green') + '\n')

    # Prompt the user for input (1 = robot wins, 2 = tie, 3 = user wins)
    print("Enter the result of this round:")
    print("1 - You wins")
    print("2 - Tie")
    print("3 - Shadow win")
    
    # Read the user's input
    user_input = input().strip()

    # Validate input
    while user_input not in ['1', '2', '3']:
        print("Invalid input. Please enter 1, 2, or 3.")
        user_input = input().strip()

    # Determine the winner based on the user's input
    if user_input == '1':
        return 0, 1  # User wins
    elif user_input == '2':
        return 0, 0  # Tie
    else:
        return 1, 0  # Robot wins


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

    # Game variables
    robot_score = 0
    user_score = 0

    while not rospy.is_shutdown():

        print("\nNew round! Let's play Rock, Paper, Scissors!")

        # Wait for the user to press Enter when they're ready
        continue_game = input("\nPress Enter when you're ready to start playing ('q' to exit)\n")
        if continue_game == 'q':
            break

        # The robot plays its move and user enters the result
        robot_points, user_points = game_round()

        # Accumulate scores
        robot_score += robot_points
        user_score += user_points

        # Display the current score
        print(f"\nCurrent Score -> Robot: {robot_score} | You: {user_score}\n")

    # End of game            
    print(f"\nThanks for playing! Final Score -> Robot: {robot_score} | You: {user_score}\n")
