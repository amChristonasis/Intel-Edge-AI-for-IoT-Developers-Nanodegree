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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.4,
                        help="Probability threshold for detections filtering"
                             "(0.4 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0
        single_image_mode = False

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        single_image_mode = False
        input_stream = args.input
        assert os.path.isfile(args.input), "file doesn't exist"

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    w = int(cap.get(3))
    h = int(cap.get(4))

    in_shape = network_shape

    #iniatilize variables
    request_id=0
    current_counter=0
    total_counter=0
    timer_enter=0
    timer_leave=0
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        net_input_image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        net_input_image = net_input_image.transpose((2, 0, 1))
        net_input_image = net_input_image.reshape(1, *net_input_image.shape)
  

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': net_input_image, 'image_info': net_input_image.shape[1:]}
#         duration_report = None
        infer_network.exec_net(request_id, net_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
         
            people_detected = 0
            probs = net_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    people_detected += 1
                    bbox = net_output[0, 0, i, 3:]
                    p1 = (int(bbox[0] * w), int(bbox[1] * h))
                    p2 = (int(bbox[2] * w), int(bbox[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
            
            # someone enters the scene
            if people_detected > current_counter:
                timer_enter = time.time()
                previous_off = timer_enter - timer_leave
                current_counter = people_detected
                if previous_off>15:
                    total_counter += people_detected - current_counter
                    client.publish("person", json.dumps({"total": total_counter}))
                                
            # someone leaves the scene
            if people_detected < current_counter:
                timer_leave = time.time()
                previous_on = timer_leave - timer_enter
                current_counter = people_detected
                if previous_on>15:
                    client.publish("person/duration", json.dumps({"duration": int(previous_on)}))
            
            client.publish("person", json.dumps({"count": current_counter}))

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()