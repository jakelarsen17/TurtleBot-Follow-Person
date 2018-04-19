# TurtleBot-Follow-Person
This repo documents our work for the ECEN 631 - Robotic Vision final project at BYU.  
Here we use person tracking from a mono camera feed to get a TurtleBot following a person.

The python script in this repo contains everything needed to run the person tracker on the TurtleBot.
Before running on the commander machine, you need to setup a ROS catkin workspace
on both the TurtleBot and the commander machine.  This script should be run from
within that workspace.  See the following repo for help on getting ROS and the 
TurtleBot set up: https://github.com/goromal/lab_turtlebot
This repo (for the image processing part of the project) is located here:
https://github.com/jakelarsen17/TurtleBot-Follow-Person

The 'MobileNetSSD' files define a trained machine learning model that can be used for a variety of 
object detection implementations.  Here we use person detection.

A 'webcam_tracker.py' script is included as a demonstration of the person tracking portion of the project.
This script should run without problems if you have a webcam on your computer and the 'MobileNetSSD' modules
in the same directory as the script (you'll need to change the paths when loading the modules in the script,
line 21).  This script does nothing with the TurtleBot.  More documentation and information on how the
tracking is done can be found at: 
https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

See the wiki attached to this repo for more detailed information on the project and how it was completed.
