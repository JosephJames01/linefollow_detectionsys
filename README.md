# 🤖Autonomous Robot for Inventory Monitoring within SMEs

This project utillsies a custom Yolov8 model trained on inventory data from a warehouse, as well as advanced navigation that utilises PID control and modelling using Matplotlib.

When combined with a custom Turtlebot-style robot, this system can successfully detect a line using the latest in computationally efficient computer vision techniques, determine 
its own position and adjust its x, y coordinates appropriately and with very high accuracy (sub 5% system overshoot and less than 1 second settling time). 

The system can then capture and report inventory data at defined positions within an environment using visual cues and further computer vision techniques. This gives the user accurate,
consistent insight of their available inventory without the need for expensive detection tech such as bardcode scanning.
