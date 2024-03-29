#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, cpu_extension=None, device='CPU'):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(self.network, device_name=device)
        layers = self.network.layers.keys()

        all_supported = True
        for l in layers:
            if l not in supported_layers:
                all_supported = False
        if not all_supported:
            print('Not all model layers supported, please specify an appropriate cpu_extension with --cpu_extension')
            self.plugin.add_extension(cpu_extension, device)
            
        self.exec_network = self.plugin.load_network(self.network, device)
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        
       
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return 
    

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
#         input_shapes = {}
#         for inp in self.network.inputs:
#             input_shapes[inp] = (self.network.inputs[inp].shape)
            
        return self.network.inputs[self.input_blob].shape
    

    def exec_net(self, request_id, net_input):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.infer_request_handle = self.exec_network.start_async(
                request_id, 
                net_input)

        return 


    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.infer_request_handle.wait()
        return status
    

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        out = self.infer_request_handle.outputs[self.output_blob]
        return out