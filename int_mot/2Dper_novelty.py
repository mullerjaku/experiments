#! /usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import random
import math
import time
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
    # Normalizes the distance based on the maximum reachable distance of the robot
    d_norm = d / AREA_MAX_DISTANCE #max=(2 x 0.85 max reach of robot) min max norm
    return d_norm

def parse_string_list(string_list): #Function for parsing the string list if we use UM
    values_list = json.loads(string_list)
    return [float(value) for value in values_list]

def novelty(d, obj_gripper, points_poses_list):
    point = np.array([d, obj_gripper])
    novelty_list = [] #List and loop for collect the distances
    for one_point in points_poses_list: #Loop for calculating the distance between the expected point and the points from the past
        d_novelty_points = np.linalg.norm(one_point - point) #Calculate distance with every point in the points_list_poses
        d_index = d_novelty_points #**2
        novelty_list.append(d_index) #List of distances between expected point and whole the points from past

    return ((sum(novelty_list)) / (len(points_poses_list)))

def neural_network(result): #Utility model for prediction
    for i, sublist in enumerate(result):
        if len(sublist) > 11:
            result[i] = sublist[-11:]

    y = np.array([])
    model = SGDRegressor()

    for i, sublist in enumerate(result):
        val = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        delka = len(sublist)
        if delka<11:
            delete = 11-delka
            val = val[delete:]
            y = np.append(y, [val])
        else: y= np.append(y, [val])

    spojeny = np.concatenate(result)
    model.fit(spojeny, y)
    return model


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
    object_pose_array = np.array([object_pose.pose.position.x, object_pose.pose.position.y, 1.12])
    
    # Add object in scene
    scene.add_object(object)
    rospy.sleep(1)

    #----HERE IS STARTING THE PROCESS-----
    t = 1
    p = 0
    loop_times = []
    c = 0.2
    v = -0.3
    b = 1.3
    while not rospy.is_shutdown():
        pose_start = geometry_msgs.msg.Pose()
        #Pose start of robot
        pose_start.position.x = c
        pose_start.position.y = v     
        pose_start.position.z = b

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
        #Start time
        start_time = rospy.Time.now().to_sec()
        #Main body of progam
        while obj_grip != 1:
            point_move_list = []
            i +=1
            print("Number of loops: ",+i)
            if i == 100:
                break

            k = 0
            while len(point_move_list) < 1000: #Loop for planning the points
                obj_grip_per = 0
                #Pose move
                pose_plan = arm_group.get_current_pose("ee_link").pose 
                vel_x = random.uniform(-0.5, 0.5) #Random move
                vel_y = random.uniform(-0.5, 0.5) #Random move
                #vel_z = random.uniform(-0.5, 0.5) #Random move
                #alpha = random.uniform(-1.5708, 1.5708) #Random rotation
                alpha = 0
                pose_plan.position.x += vel_x
                pose_plan.position.y += vel_y
                while True:
                    vel_z = random.uniform(-0.5, 0.5)  # Radnom move in z between -0.5 and 0.5
                    position_z = pose_plan.position.z + vel_z

                    if (1.0 < position_z < 1.4):
                        pose_plan.position.z = position_z
                        break

                quaternion_v2 = quaternion_from_euler(1.5708, 1.5708, alpha)
                pose_plan.orientation.x = quaternion_v2[0]
                pose_plan.orientation.y = quaternion_v2[1]
                pose_plan.orientation.z = quaternion_v2[2]
                pose_plan.orientation.w = quaternion_v2[3]
                
                #Perception of the future point
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
                
                if d_end_eff_per_norm >= 1: #Codnition for the distance and to evaluate that the robot doesnt want to go more far
                    prediction = 0
                else: 
                    prediction = novelty(d_end_eff_per_norm, obj_grip_per, points_list_poses) #Novelty value of the expected point

                #print(prediction)
                vel = [vel_x, vel_y, vel_z]
                to_list = vel + [alpha, prediction]
                point_move_list.append(to_list) #Values of every point from this exact loop
                k += 1
                arm_group.clear_pose_targets()

            #print("End of creating pose")

            #Checking if list is empty or no
            if not point_move_list:
                print("Empty list")
                continue
            
            # Sort the list by novelty value
            serazene_pole = sorted(point_move_list, key=lambda x: x[-1], reverse=True)
            vybrane_pole = serazene_pole[:10] #10 highest values

            for pole in vybrane_pole: #Try to move the robot to the expected point wit the highest novelty value
                [vel_sel_x, vel_sel_y, vel_sel_z, alpha_sel, final_d_action_sel]=pole

                #Pose move
                pose_go = arm_group.get_current_pose("ee_link").pose

                pose_go.position.x += vel_sel_x
                pose_go.position.y += vel_sel_y
                pose_go.position.z += vel_sel_z

                quaternion_v3 = quaternion_from_euler(1.5708, 1.5708, alpha_sel)
                pose_go.orientation.x = quaternion_v3[0]
                pose_go.orientation.y = quaternion_v3[1]
                pose_go.orientation.z = quaternion_v3[2]
                pose_go.orientation.w = quaternion_v3[3]

                arm_group.set_pose_target(pose_go)
                plan_success, plan, planning_time, error_code = arm_group.plan() #Just planning to see if we are in collision or no

                if not plan_success: #Action if the plan was not found
                    print("Not found plan")
                    continue

                else: #Action if the plan was found
                    arm_group.go(pose_go, wait=True)

                    #Add point real point where robot moved
                    move_pose = arm_group.get_current_pose("ee_link").pose.position
                    move_point_pose = np.array([move_pose.x, move_pose.y, move_pose.z])

                    #Distance end effector the perception
                    d_end_eff = np.linalg.norm(move_point_pose - object_pose_array)
                    norm_d = normalize(d_end_eff)
                    if norm_d < 0.025:
                        # Checking the correct position
                        obj_grip = 1

                    #Add the point in the list and delete the first point if the list is longer than 40
                    percep_point = np.array([norm_d,obj_grip])
                    points_list_poses = np.append(points_list_poses, [percep_point], axis=0) #Add the expected point in the list, not just points of this looop
                    if points_list_poses.shape[0] > 40:
                        points_list_poses = np.delete(points_list_poses, 0, axis=0)

                    perception_list.append([norm_d, obj_grip])
                    action_list.append([vel_sel_x, vel_sel_y, vel_sel_z, alpha_sel])
                    d_obj.append(norm_d)
                    obj.append(obj_grip)
                    loop_count.append(i)

                    arm_group.clear_pose_targets()
                    break

        end_time = rospy.Time.now().to_sec()
        loop_duration = end_time - start_time
        loop_times.append(loop_duration)
        
        #Plotting the graph
        plt.figure(figsize=(14, 8))
        for loop_count_val, d_val, obj_val in zip(loop_count, d_obj, obj):
            if obj_val == 0:
                marker = 'o'
            else:
                marker = 's'

            plt.plot(loop_count_val, d_val, marker=marker, markersize=8, linestyle='-', color='blue')

        plt.plot(loop_count, d_obj, linestyle='-', color='blue')

        legend_handles = [plt.Line2D([], [], marker='o', color='blue', linestyle='', label='obj_out_of_gripper'),
                            plt.Line2D([], [], marker='s', color='blue', linestyle='', label='obj_in_gripper')]

        plt.legend(handles=legend_handles)
        plt.xlabel('Time steps')
        plt.ylabel('S(distance)')
        plt.savefig(f'2D_graf_novelty_{p}.png')
        plt.close()

        p+=1
        if p == 10:
            print(loop_times)
            c = 0.15
            v = 0.15
            b = 1.15
            loop_times_1 = loop_times
            loop_times = []
        elif p == 20:
            print(loop_times)
            c = 0.4
            v = -0.8
            b = 1.35
            loop_times_2 = loop_times
            loop_times = []
        elif p == 30:
            print(loop_times_1)
            print(loop_times_2)
            print(loop_times)
            break

    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

 
