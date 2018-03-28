#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2017.12.29
# Description: collect train dataset for neural action planner.

import threading
import Queue
import pdb
import rospy
import numpy
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from image_dataset import save_dataset, DataSet
from action_status import ActionStatus

# directory for dataset file
TRAIN_DIR = '/home/blue/lab/robot/omnidirectional_vehicle/neural_planner_dataset_V2.0/tmp/train_dir/'
TRAIN_FILE = 'train-dataset-images.gz'
TRAIN_LABEL = 'train-dataset-labels.gz'

MAX_IMAGE_NUM = 500  # collect images number

IMAGE_QUEUE = Queue.Queue()  # image which receive from topic.
ACTION_QUEUE = Queue.Queue() # action which recerive from topic.


#
ACT_STATUS = ActionStatus()

class StorerThread(threading.Thread):
    """Used to store dataset to a file."""
    def __init__(self, file_dir, images_file_name, labels_file_name):
        threading.Thread.__init__(self)
        self.file_dir = file_dir
        self.images_file_name = images_file_name
        self.labels_file_name = labels_file_name
        self.bridge = CvBridge()

    def imgmsg_to_cv2(self, imgmsg):
        """Transform message 'sensor_msgs/Image' to opencv image
        object(numpy.array).
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
        except CvBridgeError as event:
            print event

        return cv_image

    def odom_to_action(self, odom):
        """Transform message 'nav_msgs/Odometry' to action label.
        'odom.twist.angular.z' is anti-clockwise direction, divided into 8 parts.
        """
        global ACT_STATUS
        vel_forward = odom.twist.twist.linear.x
        vel_rotate = odom.twist.twist.angular.z
        return ACT_STATUS.vel_to_action(vel_forward, vel_rotate)

    def run(self):
        """Acquire message from queue, and write to file.
        """
        # init
        global IMAGE_QUEUE
        global ACTION_QUEUE
        global ACT_STATUS
        global MAX_IMAGE_NUM
        demo_image = IMAGE_QUEUE.get()
	tolerance = rospy.Duration(0.5) # unit is second.
        end_flag = False
        image_num = 0
        image_rows = demo_image.height
        image_cols = demo_image.width
        image_depth = 1
        images = numpy.zeros([MAX_IMAGE_NUM, image_rows, image_cols, image_depth]\
            , dtype=numpy.uint8)
        labels = numpy.zeros([MAX_IMAGE_NUM], dtype=numpy.uint8)
        dataset = DataSet(images, labels)
        raw_image = None
        raw_action = None
        #
        while not end_flag:
            # Acquire image
            raw_image = IMAGE_QUEUE.get()
            # Find a valid action
            action_flag = True
            while action_flag:
                if raw_action == None:
                    raw_action = ACTION_QUEUE.get()

                diff = raw_image.header.stamp-raw_action.header.stamp
                if diff > rospy.Duration(0):
                    if diff < tolerance: # Time difference is small enough
                        label = self.odom_to_action(raw_action)
                        raw_action = None
                        if label != ACT_STATUS.stop: # Filter motionless image
                            print "Current Action: %d",(label)
                            break
                    else:
                        raw_action = None

                else: # action stamp is larger than image stamp
                    action_flag = False
                    break

            if not action_flag: # Can't find valid action, abort this image
                print ("warn: can't find valid action stamp, or vihecle \
			motionless, abort this image")
                continue
                
            # Tidy data
            image = self.imgmsg_to_cv2(raw_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            dataset.images[image_num, :, :, 0] = image[:, :]
            dataset.labels[image_num] = label
            # Update flags
            image_num = image_num+1
            if image_num >= MAX_IMAGE_NUM:  # Temporarily, it will be revised later
                end_flag = True
        # Save data to file
        print "Collect %d images, Begin saving..."%(image_num)
        save_dataset(self.file_dir, self.images_file_name\
            , self.labels_file_name, dataset)
        print "Save success."


LAST_IMAGE_STAMP = None
def image_callback(img):
    """This is the callback for image topic. Here, we get image, which is useful, and save
    to @IMAGE_QUEUE.
    Args:
        img: .
    Return:
        this function will write data to a global variable @IMAGE_QUEUE.
    """
    global LAST_IMAGE_STAMP
    global IMAGE_QUEUE
    peroid = rospy.Duration(0.5) # collect images periodically, The interval is 1s.
    tolerance = rospy.Duration(0.5) # tolerance is 0.5s.
    # Check image
    now = rospy.Time.now()
    if (now-img.header.stamp) > tolerance: # this image is timeout, abort it.
        return

    if (LAST_IMAGE_STAMP is not None) and (img.header.stamp-LAST_IMAGE_STAMP) \
        < peroid: # Time has not arrived yet, abort it.
        return

    LAST_IMAGE_STAMP = img.header.stamp
    # Look for action
    if IMAGE_QUEUE.put_nowait(img):
        # Failed to save
        stamp = img.header.stamp
        now = rospy.Time.now()
        print "WARN: failed to save action at stampe=%s,current stamp=%s"%(stamp, now)


def action_callback(data):
    """This is the callback for action topic.Here, we get action, and save
    to @ACTION_QUEUE.

    Args:
        data: type is 'nav_msgs/Odometry'.

    Return:
        this function will write data to a global variable @ACTION_QUEUE.
    """
    global ACTION_QUEUE
    action = data
    if ACTION_QUEUE.put_nowait(action):
        # Failed to save action
        stamp = action.header.stamp
        now = rospy.Time.now()
        print "WARN: failed to save action at stampe=%s,current stamp=%s"%(stamp, now)


def main():
    """Main"""
    rospy.init_node('collector', anonymous=True)
    store = StorerThread(TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL)
    rospy.Subscriber('/mybot/camera1/image_raw', Image, image_callback)
    rospy.Subscriber('/odom', Odometry, action_callback)
    # Launch
    store.start()
    rospy.spin()

if __name__ == '__main__':
    main()
