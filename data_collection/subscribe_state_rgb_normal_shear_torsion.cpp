#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h" 
#include "ros/time.h"
#include "geometry_msgs/WrenchStamped.h"
#include <geometry_msgs/PointStamped.h>
#include <iostream>
#include "rgb_calib_msgs/RgbCalibImgNormalShearTorsion.h"
#include "std_msgs/Float64MultiArray.h"

//void Callback_state(const std_msgs::Float64::ConstPtr & msg);
/* This function subscribes to camera, weiss and x,y,z,angle values and 
 * publishes all data in /all_data_calib_normal_shear_torsion
 */ 


rgb_calib_msgs::RgbCalibImgNormalShearTorsion global_msg_normal_shear_torsion;
int is_state_received, is_force_received, is_image_received;
rgb_calib_msgs::RgbCalibImgNormalShearTorsion global_msg_trial;
int check_init = 0;

void Callback_mode(const std_msgs::Float64 msg);
void Callback_state(const std_msgs::Float64 msg);
void Callback_pointCoor(const geometry_msgs::PointStamped msg);
void Callback_angleZ(const std_msgs::Float64 msg);
void Callback_weissforce(const geometry_msgs::WrenchStamped msg);
void Callback_camera(sensor_msgs::Image msg);

uint8_t null_data;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "subscribe_NST");  
  ros::NodeHandle n;
  ros::Subscriber sub_mode = n.subscribe("/topic_mode", 100, Callback_mode);
  ros::Subscriber sub_state = n.subscribe("/topic_state", 100, Callback_state);
  ros::Subscriber sub_pointCoor = n.subscribe("/topic_point", 100, Callback_pointCoor); 
  ros::Subscriber sub_angleZ = n.subscribe("/topic_angleZ", 100, Callback_angleZ);
  ros::Subscriber sub_weissforce = n.subscribe("/weiss_wrench", 100, Callback_weissforce);
  ros::Subscriber sub_camera = n.subscribe("/camera/image_raw", 100, Callback_camera);

  ros::Publisher all_pub = n.advertise<rgb_calib_msgs::RgbCalibImgNormalShearTorsion>("/all_data_calib_normal_shear_torsion", 2);
  ros::spinOnce();
  ros::Rate loop_rate(5);
  
  while (ros::ok())
  {
    global_msg_normal_shear_torsion.header.stamp = ros::Time::now();
    
    // publish only while pressing
    if (global_msg_normal_shear_torsion.current_state == 1.0){
      all_pub.publish(global_msg_normal_shear_torsion);
    } else { 
      ROS_ERROR("NOT PUBLISH");
    }
    ros::spinOnce();
    loop_rate.sleep();
  }
  
return 0;
}

void Callback_mode(const std_msgs::Float64 msg)
{
  global_msg_normal_shear_torsion.operation_mode = msg.data;
}

void Callback_state(const std_msgs::Float64 msg)
{
  global_msg_normal_shear_torsion.current_state = msg.data;
}

void Callback_weissforce(const geometry_msgs::WrenchStamped msg)
{
  global_msg_normal_shear_torsion.fwx = msg.wrench.torque.x;
  global_msg_normal_shear_torsion.fwy = msg.wrench.torque.y;
  global_msg_normal_shear_torsion.fwz = msg.wrench.torque.z;
  global_msg_normal_shear_torsion.fx = msg.wrench.force.x;
  global_msg_normal_shear_torsion.fy = msg.wrench.force.y;
  global_msg_normal_shear_torsion.fz = msg.wrench.force.z;
}


void Callback_camera(sensor_msgs::Image msg){
  
  /*if (check_init == 0){
    ROS_ERROR(" CHECK INITIAL FIRST TIME");
    check_init = 1;
    global_msg_trial.camera_image.data = global_msg_normal_shear_torsion.camera_image.data;
  }*/
  
  global_msg_normal_shear_torsion.camera_image.height = msg.height;
  global_msg_normal_shear_torsion.camera_image.width = msg.width;
  global_msg_normal_shear_torsion.camera_image.header = msg.header;
  global_msg_normal_shear_torsion.camera_image.encoding = msg.encoding;
  global_msg_normal_shear_torsion.camera_image.is_bigendian = msg.is_bigendian;
  global_msg_normal_shear_torsion.camera_image.step = msg.step;
  global_msg_normal_shear_torsion.camera_image.data = msg.data;

 
  /*if (global_msg_normal_shear_torsion.operation_mode == 100){
    ROS_ERROR("MAKE IMAGE NULL");
    global_msg_normal_shear_torsion.camera_image.data =  global_msg_trial.camera_image.data ;
  } else{
      ROS_ERROR("Original image");
      //global_msg_normal_shear_torsion.camera_image.data = msg.data; 
    
  }*/
}

void Callback_angleZ(const std_msgs::Float64 msg)
{
  global_msg_normal_shear_torsion.angle_z = msg.data;
}

void Callback_pointCoor(const geometry_msgs::PointStamped msg)
{
  global_msg_normal_shear_torsion.point_coordinate.x = msg.point.x;
  global_msg_normal_shear_torsion.point_coordinate.y = msg.point.y;
  global_msg_normal_shear_torsion.point_coordinate.z = msg.point.z;
}
