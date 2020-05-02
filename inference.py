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
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.model_xml = 'inference_graph.xml'
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.CPU_EXTENSION = '/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'

    def load_model(self):
        log.info('Initialize Model Loading.')
        plugin = IECore()

        log.info('Set Model Paths.')
        model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        net = IENetwork(model=self.model_xml, weights=model_bin)

        log.info('Get supported layers')
        plugin.add_extension(self.CPU_EXTENSION, "CPU")
        supported_layers = plugin.query_network(network=net, device_name="CPU")
        log.info('Supported layers length: ')
        log.info(len(supported_layers))

        log.debug('Supported layers: ')
        log.debug(supported_layers)

        unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        log.info('Unsupported layers length: ')
        log.info(len(unsupported_layers))

        log.debug('Unsupported Layers: ')
        log.debug(unsupported_layers)

        exec_net = plugin.load_network(net, "CPU")
        return exec_net



    """        
        net = IENetwork(model=self.model_xml, weights=self.model_bin)
        plugin = IEPlugin(device="CPU")
        exec_net = plugin.load(network=net)

        return exec_net
#        res = exec_net.infer({'data': img})

#>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
#>>> plugin = IEPlugin(device="CPU")
#>>> exec_net = plugin.load(network=net, num_requsts=2)
#>>> res = exec_net.infer({'data': img})



        exec_net.

        input_blob = next(iter(net.inputs))

        # Get the input shape
        input_shape = net.inputs[input_blob].shape        

        log.info("IR successfully loaded into Inference Engine.")

        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return exec_net, input_shape
    """

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return

    def exec_net(self):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return
