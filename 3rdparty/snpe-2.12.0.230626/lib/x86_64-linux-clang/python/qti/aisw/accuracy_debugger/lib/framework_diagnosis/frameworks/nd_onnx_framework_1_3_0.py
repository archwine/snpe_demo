# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.framework_diagnosis.frameworks.nd_base_framework import BaseFramework
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_warning_message, get_debugging_message
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import ModelHelper
import logging
from collections import OrderedDict

import onnx
import os
import numpy as np
from onnx import helper, shape_inference, version_converter
import onnxruntime

class OnnxFramework_1_3_0(BaseFramework):
    __VERSION__ = '1.3.0'
    FRAMEWORK_SUFFIX = '.onnx'

    def __init__(self, logger):
        super(OnnxFramework_1_3_0, self).__init__(logger)
        self._model = None
        self._graph = None
        self.ort_outs = None

    @property
    def graph(self):
        return self._graph

    def load_model(self, model_path):
        # Import graph and save to instance variable
        self._model  = onnx.load_model(model_path)
        self._graph = self._model.graph

    def add_outputs(self):
        # adds all intermediate nodes of model as output nodes
        if len(self.get_output_layers(names_only=True)) >= len(self.get_layer_identifiers()):
            # Do not modify the model if #outputnodes >= #modelnodes
            return
        for node in self._model.graph.node:
            for output in node.output:
                self._model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    def run_inference(self, input_data, input_tensor_names, output_tensor_names):
        if len(input_data) != len(input_tensor_names):
            raise FrameworkError(get_message("ERROR_FRAMEWORK_ONNX_MISMATCH_INPUTS"))
        ort_inputs = {}

        ort_session = onnxruntime.InferenceSession(self._model.SerializeToString())
        for data,input_ele in zip(input_data,ort_session.get_inputs()):
            ort_inputs[input_ele.name] = data

        outputs = [x.name for x in ort_session.get_outputs()]
        if self.ort_outs is None:
            self.ort_outs = ort_session.run(outputs, ort_inputs)

        result ={}
        for output_tensor, data in zip(outputs,self.ort_outs):
            if str(output_tensor_names[0]) == str(output_tensor):
                result.update({output_tensor: data})

        return result

    def get_intermediate_tensors(self, input_tensors, output_tensors):
        tensor_pairs = []
        input_initializer =  [node.name for node in self._model.graph.initializer]
        for node in self._model.graph.node:
            inputs = []
            outputs = []
            for input in node.input:
                input_name = onnx.ValueInfoProto(name = input).name
                if input_name not in input_initializer:
                    inputs.append(input_name)
            for output in node.output:
                outputs.append(onnx.ValueInfoProto(name=output).name)
            tensor_pairs.append((inputs, outputs))

        return tensor_pairs

    def get_dimensions(self, tensor_name):
        pass

    def get_graph_structure(self):
        """ creates a detailed list of the network's operators

        Iterates through the operators in the net, and retrieves every
        operator's index , as well as its type, inputs, and outputs

        :return: dictionary indexed by op index with values containing the
        index, tuple of list of inputs and list of outputs
        """
        op_dict = OrderedDict()
        i=0
        input_initializer =  [node.name for node in self._model.graph.initializer]
        for node in self._model.graph.node:
            inputs = []
            outputs = []
            for input in node.input:
                input_name = onnx.ValueInfoProto(name = input).name
                if input_name not in input_initializer:
                    inputs.append(input_name)
            for output in node.output:
                outputs.append(onnx.ValueInfoProto(name=output).name)
            op_dict[i]= (node.op_type, inputs, outputs)
            i +=1
        return op_dict

    def get_mapping_for_qnn_node(self, qnn_output):  # type: (str) -> str
        """ returns framework node name
        :return: the node name of qnn_output in the framework
        """
        if qnn_output[1:].isdigit():
            qnn_output = qnn_output[1:]
        check_conv_batch_norm = False
        for node in self._model.graph.node:
            if not check_conv_batch_norm:
                for output in node.output:
                    tensor_name = onnx.ValueInfoProto(name=output).name
                    tensor_replace = tensor_name.replace(".", "_")
                    tensor_replace = tensor_replace.replace("/", "_")
                    if qnn_output == tensor_replace:
                        if node.op_type == 'Conv':
                            check_conv_batch_norm = True
                            break
                        else:
                            return qnn_output
            else:
                check_conv_batch_norm = False
                if node.op_type == 'BatchNormalization':
                    return onnx.ValueInfoProto(name=node.output[0]).name # node.output[0]
                else:
                    return qnn_output

        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR")(qnn_output))
        return " "

    def get_mapping_for_snpe_node(self, snpe_output_tensor):  # type: (str) -> str
        """ returns framework node name
        :return: the node name of snpe_output_tensor in the framework
        """
        check_conv_batch_norm = False
        for node in self._model.graph.node:
            if not check_conv_batch_norm:
                for output in node.output:
                    tensor_name = onnx.ValueInfoProto(name=output).name
                    if snpe_output_tensor == tensor_name:
                        if node.op_type == 'Conv':
                            check_conv_batch_norm = True
                            break
                        else:
                            return snpe_output_tensor
            else:
                check_conv_batch_norm = False
                if node.op_type == 'BatchNormalization':
                    return onnx.ValueInfoProto(name=node.output[0]).name # node.output[0]
                else:
                    return snpe_output_tensor

        # if no matching, some warning will occur.
        logging.warning(get_warning_message("WARNING_FRAMEWORK_ONNX_MISMATCH_TENSOR")(snpe_output_tensor))
        return " "

    def get_version(self):
        return onnx.__version__

    def extract(self, start_layer_output_name, end_layer_output_name=None,
                out_model_path=None):
        raise NotImplementedError('Method extract is not implemented for onnx version < 1.8.0')

    ################################Layerwise_snooping utility methods ####################################
    def get_layer_identifiers(self, op_types_only=False):
        """
        This method returns list of layer name, output name and type in the onnx model.
        Returns:
            layers : list of tuples containing layer_name, output_name, layer_op_type.
        """
        layer_info = []
        model = self._model
        for node in model.graph.node:
            if op_types_only:
                if node.op_type not in layer_info:
                    layer_info.append(node.op_type)
            else:
                if node.op_type in ['Constant', 'Identity']:
                    continue
                layer_info.append((node.name, node.output[0], node.op_type))
        return layer_info

    def get_output_layers(self, names_only=False):
        """
        This method returns list of output layers and their datatype of provided onnx model.
        Args:
            names_only : boolean flag to return just list of output layer names
        Returns:
            output_layers_info : list of tuple containing output layer names and corresponding
            numpy datatype.
        """
        output_layers_info = []
        model = self._model

        layer_out_type_map = {}
        if not names_only:
            for node in model.graph.node:
                for idx in range(len(node.output)):
                    layer_out_type_map[node.output[idx]] = node.op_type

        # form list of tuple containing output layer names and corresponding numpy datatype
        for vi in model.graph.output:
            out_name = vi.name
            if names_only:
                output_layers_info.append(out_name)
            else:
                dim = []
                for i in range(len(vi.type.tensor_type.shape.dim)):
                    dim.append(vi.type.tensor_type.shape.dim[i].dim_value)
                try:
                    (out_dtype, _) = ModelHelper.onnx_type_to_numpy(str(vi.type.tensor_type.elem_type))
                except Exception as e:
                    logging.error(e)
                output_layers_info.append((out_name, out_dtype, layer_out_type_map[out_name], dim))
        return output_layers_info

    def get_input_layers(self, names_only=False):
        """
        This method returns list of inputs in the onnx model.
        Args:
            names_only: only return list of names
        Returns:
            input_layers_info : list of tuple containing input layer names and corresponding
            numpy datatype.
        """
        input_layers_info = []
        model = self._model
        # form list of tuple containing input layer names and corresponding numpy datatype
        for vi in model.graph.input:
            inp_name = vi.name
            if names_only:
                input_layers_info.append(inp_name)
            else:
                (inp_dtype, _) = ModelHelper.onnx_type_to_numpy(str(vi.type.tensor_type.elem_type))
                dim = []
                for i in range(len(vi.type.tensor_type.shape.dim)):
                    dim.append(vi.type.tensor_type.shape.dim[i].dim_value)
                input_layers_info.append((inp_name, inp_dtype, dim))
        return input_layers_info
