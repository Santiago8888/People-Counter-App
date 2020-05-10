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
import cv2
import time
import numpy as np

import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.model_xml = 'inference_graph.xml'
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.CPU_EXTENSION = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'

    def load_model(self):
        plugin = IECore()
        model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        net = IENetwork(model=self.model_xml, weights=model_bin)
        plugin.add_extension(self.CPU_EXTENSION, "CPU")

        supported_layers = plugin.query_network(network=net, device_name="CPU")
        unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        self.exec_net = plugin.load_network(net, "CPU") 

    def get_output(self, image):
        input_blob = next(iter(self.exec_net.inputs))
        output_blob = next(iter(self.exec_net.outputs))

        result = self.exec_net.infer({input_blob: preprocessed_image})
        result = self.exec_net.requests[0].outputs[output_blob]
        return result
