# Data Collection

## Sensor Types:

1) KMS 40 6-axis force/torque sensor from Weiss. (https://www.universal-robots.com/media/1800067/kms40_leaflet_en.pdf) 

2) Logitech, HD Pro Webcam C920. (https://www.logitech.com/en-us/product/hd-pro-webcam-c920)

3) The Universal Robots UR10. (https://wiredworkers.io/universal-robots-ur10/)


## Data Types (Recorded):

1) Force/Torque values from the sensor in 6-axis
 
2) Image from the camera

3) Robot end effector position in Cartesian Space

4) Contact type (Normal or Shear/Torsion)

4) Contact mode (Move down, Move back)


## ROS Packages :

### All data was recorded using the ROS platform. For that, the following packages/ros drivers were installed:


1) For Weiss sensor: https://github.com/ipa320/weiss_kms40

2) Camera: http://wiki.ros.org/libuvc_camera
 
3) UR robot control: https://github.com/ros-industrial/ur_modern_driver


### ROS packages were build using the CATKIN BUILD. 


## Conducting the experiments:

### Preparing the setup:

1) UR robot is started and calibrated. 

2) Run the command to start the UR robot driver: **roslaunch ur_modern_driver ur10_bringup.launch robot_ip:=192.168.1.3**

3) Start the MOVEIT: **roslaunch ur10_moveit_config ur10_moveit_planning_execution.launch** 

4) Move the robot default configurations:
	Position 1: Robot far from the sensor surface (200 mm): ** rosrun calibrate_rgb_sensor move_robot_back.py **
	Position 2: Robot close to the sensor surface (1-2 mm):** rosrun calibrate_rgb_sensor test_move_init_joints_before_calib_rgb_sensor.py **

5)Start the Weiss sensor: 
	- Provide the power supply: Voltage 24V, Current 0.1 A 
	- Run the command to start the sensor data publishing: roslaunch weiss_kms40 kms40.launch 

6)Start the camera recording: roslaunch libuvc_camera webcam_start.launch


7) Run the UR robot velocity subscriber: (Robot waits joint velocity commands to move)
	- rosrun move_ur_sim desired_velocities_node

! At this point all sensors are ready to collect the data. Next, we will be giving the robot manipulator to move in a predefined trajectory, and all data will be recorded. Image data is saved to a folder **experiment_name/image/**, while other sensor values are saved in a separate .csv file. 


### Running the setup: 

8) Move the robot manipulator: **rosrun calibrate_rgb_sensor broadcast_desired_pose_for_calib**

9) Start saving the data (images + sensor values): **rosrun calibrate_rgb_sensor subscribe_state_node**


### Remarks :


## Camera autofocus

As the camera does not have property to turn off the autofocus, while changing the setup it may be the case that camera changes focus distance and therefore image will be blured. This error is fixed with v4l2-ctl tool (http://manpages.ubuntu.com/manpages/bionic/man1/v4l2-ctl.1.html).

1) Turn on auto-focus for a while: **v4l2-ctl -d /dev/video0 --set-ctrls=focus_auto=1**

2) Find the needed focus of the camera. 

3) Turn off auto-focus: **v4l2-ctl -d /dev/video0 --set-ctrls=focus_auto=0**

