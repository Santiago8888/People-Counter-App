"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import time

import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client



def draw_boxes(frame, persons, width, height):
    for box in persons:
        xmin = int(box[3] * width)
        ymin = int(box[4] * height)
        xmax = int(box[5] * width)
        ymax = int(box[6] * height)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    return frame


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    height, width, channels = input_image.shape
    log.debug(height)
    log.debug(width)
    log.debug(channels)


    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    infer_network = Network()
    infer_network.load_model()
    prob_threshold = args.prob_threshold
    cap = cv2.VideoCapture("resources/video.mp4")

    total = 0
    frames_without_person = 6
    is_person_in_frame = False
    was_person_in_frame = False
    last_persons = []

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag: break

        image = cv2.resize(frame, (300, 300))
        preprocessed_image = preprocessing(image, 300, 300)

        result = infer_network.get_output(preprocessed_image)
        persons = [x for x in result[0][0] if x[1] == 1 and x[2] > prob_threshold]

        if persons: frames_without_person = 0


        elif frames_without_person < 6:
            persons = [x for x in result[0][0] if x[1] == 1 and x[2] > 0.25]
            if persons: frames_without_person = 0

            elif frames_without_person < 2: 
                persons = last_persons
                frames_without_person += 1

            elif frames_without_person < 12 and float(last_persons[0][3]) > 0.4 and last_persons[0][5] < 0.8:
                persons = last_persons
                frames_without_person = 0

            else: 
                frames_without_person += 1
        
        else:
                frames_without_person += 1

        is_person_in_frame = True if persons else False

        if is_person_in_frame:
            count = len(persons)
            total = count + total
            client.publish("person", json.dumps({"count": count }))

        if is_person_in_frame and not was_person_in_frame:
            start = time.time()

        if was_person_in_frame and not is_person_in_frame:
            end = time.time()
            duration = int(end - start)

            client.publish("person", json.dumps({"count": 0, "total": total }))
            client.publish("person/duration", json.dumps({"duration": duration}))

        was_person_in_frame = is_person_in_frame
        if persons: last_persons = persons


        frame = draw_boxes(frame, persons, 768, 432)
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()




        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # Grab command line args
    log.getLogger().setLevel(log.INFO)
    args = build_argparser().parse_args()

    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
