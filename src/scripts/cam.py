#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge , CvBridgeError
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from simple_pid import PID

import time
class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower', anonymous=True)
        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.camera_callback, queue_size=10)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
      
        
        self.frame_count = 0
        
       


    def camera_callback(self, data):
        # Convert the compressed image to OpenCV format
       self.frame_count += 1  # Increment frame counter
       if self.frame_count % 2 == 0:
       
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            self.show(cv_image)
        except CvBridgeError as e:
            print(e)
            return

       
        
        
    def show(self, frame): 
        cv2.imshow('frame', frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")
            return

    
if __name__ == '__main__':
    lf = LineFollower()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()