#! /usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from sklearn.linear_model import SGDRegressor
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, RobotState, PlanningScene, MoveItErrorCodes
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from colorspacious import cspace_converter
from datetime import datetime
from plyer import notification

AREA_MAX_DISTANCE = 1.2

def normalize(d):
    d_norm = d / AREA_MAX_DISTANCE #max=(2 x 0.85 max reach of robot) min max norm
    return d_norm

def inverse(database, sensory_goal):
    min_distance = float('inf')
    nearest_motor_params = None
    neighbors = []
    k = 10

    for entry in database:
        perception = np.array([entry[3], entry[4]])
        distance = np.linalg.norm(sensory_goal - perception)
        if len(neighbors) < k or distance < neighbors[-1][0]:
            # Add new neighbor and sort the array by distance
            neighbors.append((distance, entry))
            neighbors = sorted(neighbors, key=lambda x: x[0])
            # If we added more neighbors than needed, remove the most distant one
            if len(neighbors) > k:
                neighbors.pop()
    return [neighbor[1] for neighbor in neighbors] 

def main():
    rospy.init_node("robot", anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    scene.remove_world_object("my_cylinder")
    arm_name = "arm"
    hand_name = "gripper"
    arm_group = moveit_commander.MoveGroupCommander(arm_name)
    hand_group = moveit_commander.MoveGroupCommander(hand_name)
    planning_scene = PlanningScene()

    display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory,
    queue_size=20,
    )

    # Allow some leeway in position (meters) and orientation (radians)
    arm_group.set_goal_position_tolerance(0.005)
    arm_group.set_goal_orientation_tolerance(0.05)

    # Show replanning to increase the odds of a solution
    arm_group.allow_replanning(True)

    # Show 5 seconds per planning attempt
    arm_group.set_planning_time(10)
    scene.remove_attached_object("left_finger_v1_1")
    arm_group.clear_pose_targets()

    object_name = "my_cylinder"
    # Defination of object
    object_pose = PoseStamped()
    object_pose.header.frame_id = robot.get_planning_frame()
    object_pose.pose.position.x = 0.8  
    object_pose.pose.position.y = 0.1
    object_pose.pose.position.z = 0.95  

    quaternion = quaternion_from_euler(3.14, 0, 0)
    object_pose.pose.orientation.x = quaternion[0]
    object_pose.pose.orientation.y = quaternion[1]
    object_pose.pose.orientation.z = quaternion[2]
    object_pose.pose.orientation.w = quaternion[3]
    
    object_shape = SolidPrimitive()
    object_shape.type = SolidPrimitive.CYLINDER
    object_shape.dimensions = [0.04, 0.02] #The size
    
    # Create of object
    object = CollisionObject()
    object.id = object_name
    object.operation = CollisionObject.ADD
    object.header = object_pose.header
    object.primitives.append(object_shape)
    object.primitive_poses.append(object_pose.pose)
    object_pose_array = np.array([object_pose.pose.position.x, object_pose.pose.position.y, 1.12]) #1.02 is because for a moment we are checking and moving in X and Y axis 
    
    # Add object in scene
    scene.add_object(object)
    rospy.sleep(1)

    #----HERE IS STARTING THE PROCESS-----
    t = 1
    p = 9
    while not rospy.is_shutdown():
        pose_start = geometry_msgs.msg.Pose()
        #Pose start
        pose_start.position.x = 0.4
        pose_start.position.y = -0.8    
        pose_start.position.z = 1.35

        quaternion = quaternion_from_euler(1.5708, 1.5708, 0)
        pose_start.orientation.x = quaternion[0]
        pose_start.orientation.y = quaternion[1]
        pose_start.orientation.z = quaternion[2]
        pose_start.orientation.w = quaternion[3]

        arm_group.set_pose_target(pose_start)
        arm_group.go(pose_start, wait=True)
        print("Robot is in start position")
        rospy.sleep(2)
        
        #Definations
        d_obj = []
        obj = []
        action_list = []
        perception_list = []
        obj_grip = 0
        i = 0
        loop_count = []
        goals_list_d = []
        goals_list_o = []
        #First point to count the distance of poses 
        first_point = arm_group.get_current_pose("ee_link").pose.position
        rospy.sleep(1)
        first_point_pose = np.array([first_point.x, first_point.y, first_point.z])
        #Start distance
        start_dis = np.linalg.norm(first_point_pose - object_pose_array)
        start_dis_norm = normalize(start_dis)
        start_point = np.array([start_dis_norm, obj_grip])
        points_list_poses = np.empty((0, 2))
        points_list_poses = np.append(points_list_poses, [start_point], axis=0)
        norm_d = start_dis_norm

        #Main body of progam
        while obj_grip != 1:
            point_move_list = []
            i +=1
            print("Pocet opakovani: ",+i)
            if i == 100: #160
                break

            k = 0
            while len(point_move_list) < 1000: #Loop for planning the points
                obj_grip_per = 0
                #Pose move
                pose_plan = arm_group.get_current_pose("ee_link").pose 
                #rospy.sleep(1)
                vel_x = random.uniform(-0.5, 0.5) #Random move
                vel_y = random.uniform(-0.5, 0.5) #Random move
                #vel_z = random.uniform(-0.5, 0.5) #Random move
                alpha = 0
                #alpha = random.uniform(-1.5708, 1.5708) #Random rotation

                pose_plan.position.x += vel_x
                pose_plan.position.y += vel_y
                while True:
                    vel_z = random.uniform(-0.5, 0.5)  # Náhodný pohyb
                    position_z = pose_plan.position.z + vel_z

                    if (1.0 < position_z < 1.4):
                        pose_plan.position.z = position_z
                        break

                quaternion_v2 = quaternion_from_euler(1.5708, 1.5708, alpha)
                pose_plan.orientation.x = quaternion_v2[0]
                pose_plan.orientation.y = quaternion_v2[1]
                pose_plan.orientation.z = quaternion_v2[2]
                pose_plan.orientation.w = quaternion_v2[3]

                future_point_pose = np.array([pose_plan.position.x, pose_plan.position.y, pose_plan.position.z])
                theta = np.radians(alpha)
                future_point_rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                future_position = np.dot(future_point_rotation, future_point_pose) #Expected point from planning
                #print("Expected future position: ",+future_position)
                d_end_eff_fut_perception = np.linalg.norm(future_position - object_pose_array)
                d_end_eff_per_norm = normalize(d_end_eff_fut_perception)
                if d_end_eff_per_norm < 0.025:
                    # Checking the correct position
                    obj_grip_per = 1

                vel = [vel_x, vel_y, vel_z]
                to_list = vel + [d_end_eff_per_norm, obj_grip_per]
                point_move_list.append(to_list) #Values of every point from this exact loop
                
                k += 1
                arm_group.clear_pose_targets()

            print("End of creating pose")

            #Checking if list is empty or no
            if not point_move_list:
                print("Empty list")
                continue

            #Creating goal
            obj_grip_per_goal = random.choice([0, 1])
            d_per_goal = random.uniform(0, 1)
            random_goal = np.array([d_per_goal, obj_grip_per_goal])
            print("Random goal: ",+random_goal)
            near_neigh = inverse(point_move_list, random_goal)

            # for print_pole in near_neigh:
            #     print(print_pole)

            for pole in near_neigh:
                [x, y, z, dis_sel, obj_grip_sel]=pole

                #Pose move
                pose_go = arm_group.get_current_pose("ee_link").pose
                rospy.sleep(1)

                pose_go.position.x += x
                pose_go.position.y += y
                pose_go.position.z += z
                alpha_sel = 0

                quaternion_v3 = quaternion_from_euler(1.5708, 1.5708, alpha_sel)
                pose_go.orientation.x = quaternion_v3[0]
                pose_go.orientation.y = quaternion_v3[1]
                pose_go.orientation.z = quaternion_v3[2]
                pose_go.orientation.w = quaternion_v3[3]

                arm_group.set_pose_target(pose_go)
                plan_success, plan, planning_time, error_code = arm_group.plan() #Just planning to see if we are in collision or no
                rospy.sleep(2)

                if not plan_success: #Action if the plan was not found
                    print("Not found plan")
                    continue

                else:
                    #print(pole)
                    #print("Executing with curios interest value: ",+max_interest_value)
                    arm_group.go(pose_go, wait=True)

                    #Add point
                    move_pose = arm_group.get_current_pose("ee_link").pose.position
                    move_point_pose = np.array([move_pose.x, move_pose.y, move_pose.z])

                    #Distance end effector the perception
                    d_end_eff = np.linalg.norm(move_point_pose - object_pose_array)
                    norm_d = normalize(d_end_eff)
                    print("Distance is: ",+norm_d)
                    if norm_d < 0.025:
                        # Checking the correct position
                        print("The position was found")
                        obj_grip = 1

                    loop_count.append(i)
                    d_obj.append(norm_d)
                    obj.append(obj_grip)
                    goals_list_d.append(d_per_goal)
                    goals_list_o.append(obj_grip_per_goal)

                    arm_group.clear_pose_targets()
                    break

        plt.figure(figsize=(14, 8))
        for loop_count_val, d_val, obj_val in zip(loop_count, d_obj, obj):
            if obj_val == 0:
                marker = 'o'
            else:
                marker = 's'

            plt.plot(loop_count_val, d_val, marker=marker, markersize=8, linestyle='-', color='blue')

        plt.plot(loop_count, d_obj, linestyle='-', color='blue')
        for loop_count_val, goal_val_d, goal_val_o in zip(loop_count, goals_list_d, goals_list_o):
            if goal_val_o == 0:
                marker = '^'
            else:
                marker = 'D'

            plt.scatter(loop_count_val, goal_val_d, marker=marker, s=40, color='red')

        legend_handles = [plt.Line2D([], [], marker='o', color='blue', linestyle='', label='obj_out_of_gripper'),
                            plt.Line2D([], [], marker='s', color='blue', linestyle='', label='obj_in_gripper'),
                            plt.Line2D([], [], marker='^', color='red', linestyle='', label='curiosity_goal_obj_out'),
                            plt.Line2D([], [], marker='D', color='red', linestyle='', label='curiosity_goal_obj_in')]

        plt.legend(handles=legend_handles)
        plt.xlabel('Time steps')
        plt.ylabel('S(distance)')
        plt.savefig(f'2D_graf_{p}.png')

        p+=1
        if p == 10:
            break

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

 
