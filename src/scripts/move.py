#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import numpy as np
from simple_pid import PID
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import matplotlib
import pickle


class RobotController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('robot_controller', anonymous=True)
        # Set up the subscriber for compressed images
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.image_callback)
        # Set up the publisher for velocity commands
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pid = PID(0.003, 0.0035, 0.0006, setpoint=0)
        self.pid.output_limits = (-0.3, 0.3) 
        # Set up the CvBridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        # Set the rate at which to publish velocity commands
        self.rate = rospy.Rate(10)  # 10Hz
        self.trigger = False
        self.angular_velocity =0
        self.line =0
        self.position=0
        self.vid = 0
        self.errors =  []
        self.times =[]
        self.start_time = rospy.get_time()
        self.frame_count =0

    
    def image_callback(self, data):
       self.frame_count += 1  # Increment frame counter
       if self.frame_count % 4 == 0:
        try:
            # Convert the ROS image to OpenCV format
            frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            height, width = frame.shape[:2]
           
            if self.vid == 10:  
              
              now = self.position
              outputposition  = 'outputposition' + str(now)

              success =  cv2.imwrite(f'results/{outputposition}.jpg', frame)
            
              if success:
                 rospy.loginfo("The image was successfully saved")
              #else:
                 #rospy.loginfo("Error: The image could not be saved.")
            
               
            lower_half = frame[int(3*height/4):, :]  # Focus on the bottom quarter
            lower_half_processed = self.preprocess_image(lower_half)
           
            contours1, _ = cv2.findContours(lower_half_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            filtered_contours = []
          
            for contour1 in contours1:
             x, y, w, h = cv2.boundingRect(contour1)
             
             if  w*h > 300*40:
                
                filtered_contours.append(contour1)
                aspect_ratio = float(w)/h
                if aspect_ratio > 6 and w*h > 20000: 
                 self.line = self.line +1
                 
                if self.line == 1: 
                  self.trigger = True
                  rospy.loginfo("True")

            if filtered_contours:
             largest_contour = max(filtered_contours, key=cv2.contourArea)
         
             self.draw_bounding_boxes([largest_contour], lower_half, (0, 255, 0))  # Use green for visibility
             
             center_of_box = x + w/ 2
             center_of_image = frame.shape[1] / 2
             error = center_of_image - center_of_box
             self.angular_velocity = -self.pid(error)
             
             current_time = rospy.get_time() - self.start_time
             self.errors.append(error)
             self.times.append(current_time)
             

            cv2.imshow("Camera View", frame)
            cv2.imshow('processed', lower_half_processed)
       
            cv2.waitKey(1)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    
    def preprocess_image(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)  #  Gaussian Blur 5x5 Kernel to keep some image sharpness
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert to HSV
        Black_ll = np.array([0, 0, 0], np.uint8)       # Black Lower limit set to 0
        Black_Ul = np.array([180, 255, 100], np.uint8)   # Black upper limit 
        Black_mask = cv2.inRange(hsv, Black_ll, Black_Ul)  # Black mask 
        kernel = np.ones((5,5), np.uint8)                  # 5x5 kernel for small imperfections and incomplete boundaries
        closing = cv2.morphologyEx(Black_mask, cv2.MORPH_CLOSE, kernel)
        return closing
    
    
    def draw_bounding_boxes(self, contours, image, color):
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    
    def publish_velocity(self, angular, duration):
        # Create a Twist message for the velocity command
         twist = Twist()
         twist.angular.z = angular
        
        # Calculate the end time
         end_time = rospy.Time.now() + rospy.Duration(duration)
        
        # Loop until the duration is reached
         while rospy.Time.now() < end_time:
            # Publish the Twist message
            self.vel_pub.publish(twist)
            self.rate.sleep()
        
        
         self.stop

          
    def capture(self, frame):
     
      now = self.position
      outputposition  = 'outputposition' + str(now)
      try:
       cv2.imwrite('outputposition.jpg', frame)
      except Exception as e:
        rospy.loginfo(f"Error writing file: {e}")
      
    
    def publish_velocitylin(self, lin, duration):
        # Create a Twist message for the velocity command
         twist = Twist()
         twist.linear.x = lin
        
        # Calculate the end time
         end_time = rospy.Time.now() + rospy.Duration(duration)
        
        # Loop until the duration is reached
         while rospy.Time.now() < end_time:
            # Publish the Twist message
            self.vel_pub.publish(twist)
            self.rate.sleep()
        
        
         self.stop

    def stop(self):
        # Create a Twist message with zero velocity
        twist = Twist()
        self.vel_pub.publish(twist)
        self.rate.sleep()

    def run(self):
        # Main loop
       while not rospy.is_shutdown():
           if self.trigger:
            self.stop
            self.position = self.position + 1
            time = 4.5
            angular_velocityp= 0.3 
            angular_velocityn = -0.3

            end_time0 = rospy.Time.now() + rospy.Duration(6)
            
            self.publish_velocity(angular_velocityp,time)
            
            end_time = rospy.Time.now() + rospy.Duration(5)
            while rospy.Time.now() < end_time:
                self.vid = self.vid +1
                self.rate.sleep()
                 
            self.publish_velocity(angular_velocityn, time)
            
            end_time2 = rospy.Time.now() + rospy.Duration(1)
            while rospy.Time.now() < end_time2:
                self.rate.sleep()

            self.publish_velocitylin(0.1,1)
           
            self.end()
            self.line = 0 
            continue
           else: 
               
               self.send_velocity(0.04, self.angular_velocity)
               with open('data.pkl', 'wb') as f:
                pickle.dump((self.errors, self.times), f)
              
              

    def end(self): 
     self.trigger = False

    def send_velocity(self, linear_x, angular_z):
        twist = Twist()
        twist.angular.z = angular_z
        twist.linear.x = linear_x
        
        self.vel_pub.publish(twist)
        self.rate.sleep()
         
if __name__ == '__main__':
    try:
        # Create an instance of the RobotController
        controller = RobotController()
        controller.run()
        
    except rospy.ROSInterruptException:
        pass
    finally:
        # Close the OpenCV window
        cv2.destroyAllWindows()