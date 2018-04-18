# TurtleBot-Follow-Person
This repo documents our work for the ECEN 631 - Robotic Vision final project.  
Here we use person tracking from a mono camera feed to get a TurtleBot following a person.

The python script in this repo contains everything needed to run the person tracker on the TurtleBot.
Before running on the commander machine, you need to setup a ROS catkin workspace
on both the TurtleBot and the commander machine.  This script should be run from
within that workspace.  See the following repo for help on getting ROS and the 
TurtleBot set up: https://github.com/goromal/lab_turtlebot
This repo (for the image processing part of the project) is located here:
https://github.com/jakelarsen17/TurtleBot-Follow-Person
