import time
import roslib; roslib.load_manifest('ur_driver')
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

Q1 = [-1.542433526963542, -2.0589992756930044, -1.5388309922562193,
        -1.116094317097714, 1.570833890149508, -3.113168325148432]

client = None

def move1():
    g = FollowJointTrajectoryGoal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    joint_states = rospy.wait_for_message("joint_states", JointState)
    joints_pos = joint_states.position
    g.trajectory.points = [ JointTrajectoryPoint(positions=joints_pos, velocities=[0]*6,
        time_from_start=rospy.Duration(0.0)), JointTrajectoryPoint(positions=Q1,
        velocities=[0]*6, time_from_start=rospy.Duration(5.0))]
    
    client.send_goal(g)
    try:
        client.wait_for_result()
    except KeyboardInterrupt:
        client.cancel_goal()
        raise
   

def main():
    global client
    try:
        rospy.init_node("test_move", anonymous=True, disable_signals=True)
        client = actionlib.SimpleActionClient('follow_joint_trajectory', FollowJointTrajectoryAction)
        print "Waiting for server..."
        client.wait_for_server()
        print "Connected to server"
        move1()
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        raise

if __name__ == '__main__':
    main()
