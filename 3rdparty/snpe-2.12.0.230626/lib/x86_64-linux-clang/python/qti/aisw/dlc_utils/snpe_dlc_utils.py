#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2018-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from __future__ import print_function

import os
import zipfile
from functools import reduce
from collections import OrderedDict
import csv
import logging
import sys
import math
import numpy as np

try:
    from qti.aisw.dlc_utils import modeltools
    from qti.aisw.converters.common import ir_graph
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

csv_file_flag = False

# sizes of various C data types, used for memory estimation
SIZEOF_INT8_T = 1
SIZEOF_UINT8_T = 1
SIZEOF_INT16_T = 2
SIZEOF_HALF = 2
SIZEOF_FLOAT = 4

def setUpLogger(verbose):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(lvl)
    formatter = logging.Formatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def color_space_name(type_code):
    # tracks the DNN PB format
    cs_name = { 0:"rgb",
                1:"argb32",
                2:"rgba",
                3:"nv21",
                4:"bgr",
                5:"blob1d",
                6:"blob2d" }

    return cs_name[type_code]

def input_type_name(type_code):
    # tracks the DNNPB format
    type_name = { 0:"default",
                  1:"image" ,
                  2:"opaque"}
    return type_name[type_code]

def padding_mode_name(type_code):
    mode_name = { 1:"zero",
                  2:"reflect",
                  3:"constant",
                  4:"edge"}

    return mode_name[type_code]

def resize_mode_name(type_code):
    resize_mode = { 0:"bilinear",
                    1:"nearest_neighbor" }
    return resize_mode[type_code]

def prior_box_name(type_code):
    prior_box_type = { 0:"corner",
                       1:"center_size",
                       2:"corner_size" }
    return prior_box_type[type_code]

def nms_type_name(nms_type_code):
    nms_type = { 0:"fast",
                 1:"regular" }
    return nms_type[nms_type_code]

def print_row(values, col_sizes, csv_content = []):
    str = '| '
    for value, size in zip(values, col_sizes):
        str += '{0:<{1}}|'.format(value, size) + ' '
    str = str[:-1]
    print (str)
    if csv_file_flag:
        csv_content.append(values)

def print_value(value, csv_content = []):
    print(value)
    if csv_file_flag:
        for i in value.split('\n'):
            csv_content.append([i])

def print_all(print_list, col_sizes):
    for list in print_list:
        list.append(col_sizes)
        print_row(list[0], list[2], list[1])

def print_headers(headers, col_sizes, csv_content, total_size):
    print_value('-'*total_size)
    print_row(headers, col_sizes, csv_content)
    print_value('-'*total_size)

def product(numbers):
    if len(numbers) == 0:
        return 1
    else:
        return reduce((lambda x, y: x * y), numbers)

def get_si_notation(n, total):
    if (total > 0):
        percent = 100*float(n)/total
    else:
        percent = 0
    if n < 1000:
        return "%d (%.3g%%)" % (n, percent)
    elif n < 1000*1000:
        return '%dk (%.3g%%)' % (n/1000, percent)
    else:
        return '%dM (%.3g%%)' % (n/(1000*1000), percent)

def get_binary_si_notation(n, total):
    result_string = ""

    if n < 1024:
        result_string = "%d B" % (n)
    elif n < 1024*1024:
        result_string = '%.01f kiB' % (n/1024.0)
    else:
        result_string = '%.01f MiB' % (n/(1024.0*1024.0))

    if total is not None:
        if (total > 0):
            percent = 100*float(n)/total
        else:
            percent = 0
        if n < 1024:
            result_string += "(%.2f%%)" % (percent)
        elif n < 1024*1024:
            result_string += '(%.2f%%)' % (percent)
        else:
            result_string += '(%.2f%%)' % (percent)

    return result_string

def is_scale_offset_encoding(encType):
    try:
        if encType == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET or \
                encType == ir_graph.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET :
            return True
    except AttributeError:
        pass

    return False

def is_axis_scale_offset_encoding(encType):
    try:
        if encType == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET or \
                encType == ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET :
            return True
    except AttributeError:
        pass

    return False

class OpRow (object):
    def __init__(self, op, prev_rows, show_all_encodings=False):
        self.op = op
        self.id = len(prev_rows)
        self.name = op.name
        self.type = op.type
        self.show_all_encodings = show_all_encodings

        # Append a string representation of dataType to the tensor name
        def tensor_name_and_type(t):
            dimension = '['
            for dim in t.dims():
                dimension += str(dim) + ', '
            dimension = dimension[:-2] + ']'
            return t.name() + ' (' + 'data type: ' + t.data_type_string() \
                   + '; ' + 'tensor dimension: ' + dimension + '; ' + 'tensor type: ' + t.tensor_type() + ')'

        def nw_input_marker(t):
            if t.is_app_write_tensor():
                return ' [NW Input]'
            return ''

        self.input_names = [input_tensor.name() for input_tensor in op.inputs()]
        self.input_names_and_types = [tensor_name_and_type(t) + nw_input_marker(t) for t in op.inputs()]

        self.input_dims = op.get_input_shapes()
        if len(self.input_dims) < 1:
            self.input_batch = 0
        elif len(self.input_dims[0]) < 1:
            self.input_batch = 0
        else:
            self.input_batch = self.input_dims[0][0]
        self.output_names = []
        self.output_names_and_types = []

        self.output_dims_list = []
        self.output_precisions = []
        self.memory_fxp = 0
        self.memory_cpu = 0
        for output_tensor in op.outputs():
            self.output_names.append(output_tensor.name())
            self.output_names_and_types.append(tensor_name_and_type(output_tensor))

            self.output_dims_list.append(output_tensor.dims())
            self.output_precisions.append(output_tensor.data_type())
            self.memory_cpu += self.calc_alignedbuffer_size(product(output_tensor.dims()), SIZEOF_FLOAT)
            self.memory_fxp += self.calc_alignedbuffer_size(product(output_tensor.dims()), SIZEOF_UINT8_T)

        # TODO: Add this back in when Qnn validation is enabled
        # self.valid_cpu = layer['valid_cpu']
        # self.valid_gpu = layer['valid_gpu']
        # self.valid_dsp = layer['valid_dsp']
        # self.valid_aip = layer['valid_aip']

        self.macs = 0
        self.param_count = 0 # i.e. size of weights
        self.nativeDimSize = 3
        self.op_affinity = op.get_op_affinity()

        self.parms = []

        self.memory_cpu = 0
        self.memory_fxp = 0

        # Encoding info by tensor name
        self.encoding_infos = {}

        self.layer_affinity = op.get_op_affinity_string()

        # TODO: We dont support fxp data on CPU in Qnn backends
        # every layer uses at least the memory for its own output buffers
        # for output_index in range(0, len(self.output_dims_list)):
        #
        #     output_dims = self.output_dims_list[output_index]
        #
        #     if self.output_precisions[output_index] == modeltools.PRECISION_FIXED_8:
        #         cpu_element_size = SIZEOF_INT8_T
        #     else:
        #         cpu_element_size = SIZEOF_FLOAT

        # encoding_type = model.get_tf_encoding_type()
        # if encoding_type == 'QMN':
        #     format_func = lambda encoding: "Q%d.%d" % encoding
        #     output_encoding = model.get_fxp_output_encoding(self.name)
        #     if output_encoding is not None:
        #         self.add_parm("output encoding", format_func(output_encoding))
        #
        #     weight_extract_func = model.get_fxp_weight_encoding

        #TODO: Add static input tensor fxp data extractors
        scale_offset_format_func = lambda encoding: "bitwidth %d, min %.12f, max %.12f, scale %.12f, offset %.12f" % encoding
        axis_scale_offset_format_func = lambda axis, num_elements: "axis: %d, num_elements: %d" % (axis, num_elements) \
                if (num_elements == 1 or self.show_all_encodings) else "axis: %d, num_elements: %d (above encoding is only for the first (channel_0) of %d channels)" \
                % (axis, num_elements, num_elements)

        # Helper for calls to self.add_parm with encoding information
        def add_encoding_parms(name, encoding):
            def make_list_of_encoding_tuple(enc):
                def make_encoding_tuple_(encinfo):
                    return encinfo.bw, encinfo.min, encinfo.max, encinfo.scale, encinfo.offset

                if is_scale_offset_encoding(enc.type) :
                    return [make_encoding_tuple_(enc.encInfo)]
                elif is_axis_scale_offset_encoding(enc.type) :
                    list_of_encInfos = [make_encoding_tuple_(x) for x in enc.axisEncInfo.encInfos]
                    return list_of_encInfos

                return None

            list_of_encoding_tuple = make_list_of_encoding_tuple(encoding)
            if list_of_encoding_tuple:
                key = name + " encoding"
                encinfo_str = scale_offset_format_func(list_of_encoding_tuple[0])
                self.encoding_infos[name] = encinfo_str
                if is_scale_offset_encoding(encoding.type) :
                    self.add_parm(key + ' ', encinfo_str)
                elif is_axis_scale_offset_encoding(encoding.type) :
                    self.add_parm(key + ' for channel_0', encinfo_str)
                    if(self.show_all_encodings):
                        for i in range(1,len(list_of_encoding_tuple)):
                            encinfo_str = scale_offset_format_func(list_of_encoding_tuple[i])
                            self.add_parm(key + ' for channel_' + str(i), encinfo_str)
                            self.encoding_infos[name] = encinfo_str
                    aq_str = axis_scale_offset_format_func(encoding.axisEncInfo.axis,
                            len(encoding.axisEncInfo.encInfos))
                    self.add_parm("axis-quant", aq_str)
                    self.encoding_infos[name] += ' ' + aq_str

        # Add APP_WRITE tensors encoding info
        for input in op.inputs():
            if input.is_app_write_tensor():
                add_encoding_parms(input.name(), input.get_encoding())

        # Add static tensor encoding info
        for input in op.inputs():
            if input.is_static_tensor():
                add_encoding_parms(input.name(), input.get_encoding())

        # Add output tensor encoding info
        for output in op.outputs():
            add_encoding_parms(output.name(), output.get_encoding())


        self.extract_op_attrs(op)

        def extract_noop(op):
            pass
        extractor = getattr(self, 'extract_%s' % op.type, extract_noop)
        extractor(op)

    # Method to combine rows' encoding info dicts for network input summary table
    def merge_encoding_infos(self, encinfo_dict):
        for k in self.encoding_infos:
            encinfo_dict[k] = self.encoding_infos[k]

    def dump(self, col_sizes, csv_content):
        print_row([str(self.id), self.name, self.type, self.get_input_name_and_type(0),
                   self.get_output_name_and_type(0), self.outputs_string(0), self.get_runtimes(), self.get_parm(0)],
                  col_sizes, csv_content)

        extra_rows = max(len(self.input_names), len(self.parms))
        extra_rows = max(extra_rows, len(self.output_names))

        for i in range(1,extra_rows):
            print_row(["", "", "", self.get_input_name_and_type(i), self.get_output_name_and_type(i),
                       self.outputs_string(i), "", self.get_parm(i)], col_sizes, csv_content)

    def outputs_string(self, idx):
        if idx >= len(self.output_dims_list):
            return ""
        else:
            return 'x'.join(map(str, self.output_dims_list[idx]))

    # calculate the size in memory that an alignedbuffer would need to be in order to
    # hold the given number of elements
    def calc_alignedbuffer_size(self, num, size, alignment=16):
        return (num + alignment) * size; # AlignedBuffer always allocs an extra 16 elements for alignment

    def id(self):
        return self.id

    def id_width(self):
        return len(str(self.id))

    def name(self):
        return self.name

    def name_width(self):
        return len(self.name)

    def type(self):
        return self.type

    def type_width(self):
        return len(self.type)

    def input_names(self):
        return self.input_names

    def input_width(self):
        return max(list(map(len,self.input_names))+[0])

    def input_name_and_type_width(self):
        return max(list(map(len,self.input_names_and_types))+[0])

    def output_names(self):
        return self.output_names

    def output_width(self):
        return max(list(map(len,self.output_names))+[0])

    def output_name_and_type_width(self):
        return max(list(map(len,self.output_names_and_types))+[0])

    def output_dims_width(self):
        return len(self.outputs_string(0))

    def parms_width(self):
        return max(list(map(len, self.parms))+[0])

    def runtimes_width(self):
        return len(self.get_runtimes())

    def get_parm(self, i):
        if i >= len(self.parms):
            return ""
        else:
            return self.parms[i]

    def get_parm_list(self):
        return self.parms

    def get_input(self,i):
        if i >= len(self.input_names):
            return ""
        else:
            return self.input_names[i]

    def get_input_name_and_type(self,i):
        if i >= len(self.input_names_and_types):
            return ""
        else:
            return self.input_names_and_types[i]

    def get_input_name_and_dimension(self,i):
        if i >= len(self.input_names_and_types):
            return ""
        else:
            sep_in = '; tensor type:'
            temp_in = self.input_names_and_types[i]
            stripped_in = temp_in.split(sep_in, 1)[0] + ')'
            return stripped_in

    def get_input_list(self):
        return self.input_names

    def get_output(self,i):
        if i >= len(self.output_names):
            return ""
        else:
            return self.output_names[i]

    def get_output_name_and_type(self,i):
        if i >= len(self.output_names_and_types):
            return ""
        else:
            return self.output_names_and_types[i]

    def get_output_name_and_dimension(self,i):
        if i >= len(self.output_names_and_types):
            return ""
        else:
            sep_out = '; tensor type:'
            temp_out = self.output_names_and_types[i]
            stripped_out = temp_out.split(sep_out, 1)[0] + ')'
            return stripped_out

    def get_output_list(self):
        return self.output_names

    def get_runtimes(self):
        #Todo: Determine what to do with these. Will probably need to validate ops in
        #      IrGraph
        runtimes_str = ""
        runtimes_str += "A "
        runtimes_str += "D "
        runtimes_str += "G "
        runtimes_str += "C"
        return runtimes_str

    def get_num_params(self):
        return self.param_count

    def get_macs(self):
        return self.macs

    # Gets the amount of memory needed to set up the op in bytes
    def get_memory_cpu(self):
        return self.memory_cpu

    def get_memory_fxp(self):
        return self.memory_fxp

    def add_parm( self, key, val ):
        if type(val) is float:
            valstring = "%.4g" % val
        else:
            valstring = str(val)
        self.parms.append("%s: %s" % (key, valstring))

    def extract_op_attrs(self, op):
        attr_names = op.attrs.list_names()
        for name in attr_names:
            #parse scalar attrs
            if (name == "op_affinity"):
                continue
            elif op.attrs.get_attr_type(name) == ir_graph.QNN_PARAMTYPE_SCALAR:
                if op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_8 or \
                   op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_SFIXED_POINT_8:
                    self.add_parm(name, op.attrs.get_int8(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_16 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_SFIXED_POINT_16:
                    self.add_parm(name, op.attrs.get_int16(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_32 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_SFIXED_POINT_32:
                    self.add_parm(name, op.attrs.get_int32(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_64:
                    self.add_parm(name, op.attrs.get_int64(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_8 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UFIXED_POINT_8:
                    self.add_parm(name, op.attrs.get_uint8(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_16 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UFIXED_POINT_16:
                    self.add_parm(name, op.attrs.get_uint16(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_32 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UFIXED_POINT_32:
                    self.add_parm(name, op.attrs.get_uint32(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_64:
                    self.add_parm(name, op.attrs.get_uint64(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_FLOAT_16 or \
                    op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_FLOAT_32:
                    self.add_parm(name, op.attrs.get_float(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_BOOL_8:
                    self.add_parm(name, op.attrs.get_bool(name))
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UNDEFINED:
                    self.add_parm(name, op.attrs.get_string(name))
            elif op.attrs.get_attr_type(name) == ir_graph.QNN_PARAMTYPE_TENSOR:
                # Parse Tensor Attributes
                tensor_data = op.attrs.get_static_tensor_data(name)
                if tensor_data.size > 10:
                    continue
                str_array = np.array2string(tensor_data, separator=',').replace('\n', '') \
                                 .replace(' ', '').replace(',', ', ')
                self.add_parm(name, str_array)

    def extract_Batchnorm(self, op):
        weights_channel = op.get_input_shapes()[1][0]

        self.param_count = weights_channel * 2  # weights + bias (same shape of channel for each)
        input_tensor_size = len(self.input_dims[0])
        if(input_tensor_size == 1 or input_tensor_size == 2):
            self.nativeDimSize = 1
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

        self.memory_cpu += self.calc_alignedbuffer_size(weights_channel, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights_channel, SIZEOF_UINT8_T)

    def extract_LayerNorm(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims) * 5  # 5 to account for mu and sigma calculation

        weights_shape = op.get_input_shapes()[1]
        self.memory_cpu += self.calc_alignedbuffer_size(weights_shape[0], SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights_shape[0], SIZEOF_UINT8_T)

    def extract_cmrn(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)*3

    def extract_Conv2d(self, op):
        weights_shape = op.get_input_shapes()[1]
        self.param_count = product(weights_shape)
        self.param_count += weights_shape[3] # biases
        native_input_dims = self.input_dims[0][-self.nativeDimSize:]
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]

        # calculate size of expanded buffer
        conv_image_width = weights_shape[0] * weights_shape[1] * native_input_dims[2]
        rounding_amount = 32;
        expanded_stride = ((conv_image_width + rounding_amount - 1) / rounding_amount) * rounding_amount;
        expanded_height = native_output_dims[0] * native_output_dims[1];
        expanded_buffer_size = expanded_height * expanded_stride

        # calculate size of weight array
        weights_size = product(weights_shape)

        # = filter size * number of filter positions
        self.macs = product(weights_shape[0:3])*product(native_output_dims)

        self.memory_cpu += self.calc_alignedbuffer_size(expanded_buffer_size, SIZEOF_FLOAT) + self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(expanded_buffer_size, SIZEOF_UINT8_T) + self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)

    def extract_TransposeConv2d(self, op):
        self.extract_Conv2d(op)

        # for deconvolution, macs are computed off number of input positions.
        native_input_dims = self.input_dims[0][-self.nativeDimSize:]
        input_size = product(native_input_dims)
        weight_dim = op.get_input_shapes()[1]

        weights_size = product(weight_dim[0:4])

        group = op.attrs.get_uint32('group') if op.attrs.has('group') else  1
        self.macs = weight_dim[0]*weight_dim[1]*weight_dim[3]*input_size/group

        self.memory_cpu += self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)
        self.param_count = weights_size + (weight_dim[3] * group)  # weights_shape + bias_shape

    def extract_ElementWiseAdd(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseAnd(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseDivide(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseEqual(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseFloorDiv(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseGreater(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseGreaterEqual(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseLess(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseLessEqual(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseMaximum(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseMinimum(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseMultiply(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseOr(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWisePower(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseSelect(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseSquaredDifference(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ElementWiseSubtract(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_FullyConnected(self, op):
        if len(op.inputs()) == 3:
            self.param_count = op.get_input_shapes()[2][0]

        weights_size = product(op.get_input_shapes()[1])

        self.param_count += weights_size
        self.macs += weights_size
        self.memory_cpu += self.calc_alignedbuffer_size(weights_size, SIZEOF_FLOAT)
        self.memory_fxp += self.calc_alignedbuffer_size(weights_size, SIZEOF_UINT8_T)

    def extract_L2Norm(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims) * 2  # 2 since summation is squared

    def extract_MatMul(self, op):
        #TODO: Not part of opDef?
        #self.param_count = layer['bias'].shape[0]
        # macs is m * n * k, where m and n are outer dims after transpose and k is the inner common dim
        k = self.input_dims[0][-1]
        if op.attrs.has('transpose_in0'):
            if op.attrs.get_bool('transpose_in0'):
                k = self.input_dims[0][-2]
        self.macs = product(self.input_dims[0][:-2]) * product(self.input_dims[1][:-2]) * k

    def extract_Lrn(self, op):
        self.extract_cmrn(op)
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        #Window size = 2* radius
        self.macs = product(native_output_dims)*2*op.attrs.get_int32('radius')*2*op.attrs.get_int32('radius')

# TODO: Enable Lstm
#    def extract_lstm(self, op):
#         self.add_parm("x weights", op.get_input_shapes()[1])
#         self.param_count += product(op.get_input_shapes()[1])
#         self.add_parm("x weights", op.get_input_shapes()[5])
#         self.param_count += product(op.get_input_shapes()[5])
#         self.param_count += product(op.get_input_shapes()[8])
#         self.add_parm("biases", op.get_input_shapes()[8])
#
#         if 'x_static_gate_weights' in layer:
#             x_static_weights = layer['x_static_gate_weights']
#             self.param_count += product(x_static_weights.shape)
#             self.add_parm("x_static weights", x_static_weights.shape)
#
#         self.add_parm("backward",  layer['backward'])
#         input_features = self.input_dims[0][-1]
#         output_features = self.output_dims_list[0][-1]
#         steps = self.output_dims_list[0][-2]
#         self.macs = 4*input_features*output_features*steps
#         self.macs += 4*output_features*output_features*steps
#         self.macs += 3*output_features*steps
#
#         if 'reset_state_at_time_step_0' in layer:
#             self.add_parm("reset_state_at_time_step_0", layer['reset_state_at_time_step_0'])
#
#         if 'peephole_weights' in layer:
#             peephole_weights = layer['peephole_weights']
#             self.param_count += product(peephole_weights.shape)
#             self.add_parm("peephole weights", peephole_weights.shape)
#
#         if 'cell_clip' in layer:
#             self.add_parm("cell clip", layer['cell_clip'])
#
#         if 'projection_weights' in layer:
#             projection_weights = layer['projection_weights']
#             self.param_count += product(projection_weights.shape)
#             self.add_parm("projection weights", projection_weights.shape)
#         if 'projection_biases' in layer:
#             projection_biases = layer['projection_biases']
#             self.param_count += product(projection_biases.shape)
#             self.add_parm("projection biases", projection_biases.shape)
#         if 'projection_clip' in layer:
#             self.add_parm("projection clip", layer['projection_clip'])
#
#         if 'normalization_weights' in layer:
#             normalization_weights = layer['normalization_weights']
#             self.param_count += product(normalization_weights.shape)
#             self.add_parm("normalization weights", normalization_weights.shape)
#         if 'epsilon' in layer:
#             self.add_parm("epsilon", layer['epsilon'])
#         if 'input_gate_qscale' in layer:
#             self.add_parm("input gate qscale", layer['input_gate_qscale'])
#         if 'forget_gate_qscale' in layer:
#             self.add_parm("forget gate qscale", layer['forget_gate_qscale'])
#         if 'cell_gate_qscale' in layer:
#             self.add_parm("cell gate qscale", layer['cell_gate_qscale'])
#         if 'output_gate_qscale' in layer:
#             self.add_parm("output gate qscale", layer['output_gate_qscale'])
#         if 'hidden_state_offset' in layer:
#             self.add_parm("hidden state offset", layer['hidden_state_offset'])
#         if 'hidden_state_qscale' in layer:
#             self.add_parm("hidden state qscale", layer['hidden_state_qscale'])

    def extract_PoolAvg2d(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        filter_size_tensor = op.attrs.get_static_tensor_data('filter_size')
        self.macs = product(native_output_dims) * product(filter_size_tensor)

    def extract_PoolAvg3d(self, op):
        self.extract_PoolAvg2d(op)

    def extract_Prelu(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)
        self.param_count = product(op.get_input_shapes()[1])


    def extract_RoiAlign(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_RoiPooling(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_ResizeBilinear(self, op):
        native_output_dims = self.output_dims_list[0][-self.nativeDimSize:]
        self.macs = product(native_output_dims)

    def extract_Transpose(self, op):
        native_output_dims = self.output_dims_list[0][:]


class ModelInfo(object):
    def __init__(self):
        self.model_reader = modeltools.IrDlcReader()
        self.cache_reader = modeltools.IrDlcCacheRecordReader()
        self.graph = None
        self.model_filename = ""
        self.rows = []
        self.op_mapping = {}
        self.total_params = 0
        self.total_macs = 0
        self.total_memory_cpu = 0
        self.total_memory_fxp = 0
        self.total_op_types = set()

    def load(self, input_file_name):
        self.model_reader.open(input_file_name)
        self.model_filename = input_file_name

    def extract_model_info(self, input_file_name, show_all_encodings=False):
        self.load(input_file_name)

        self.graph = self.model_reader.get_ir_graph()
        for op in self.graph.get_ops():
            row = OpRow(op, self.rows, show_all_encodings)
            self.rows.append(row)
            self.total_params += row.get_num_params()
            self.total_macs += row.get_macs()
            self.total_memory_cpu += row.get_memory_cpu()
            self.total_memory_fxp += row.get_memory_fxp()
            self.total_op_types.add(str(op.type))

        return self.rows

    def dump_info(self, show_memory_usage, input_file_name, output_file_name, show_all_encodings=False):

        """
        Dump information about the given DLC file.

        If output_file_name is not None, then the info is written as CSV to that file.
        If it is none, then the dump is printed to the screen.
        """

        self.extract_model_info(input_file_name, show_all_encodings)

        global csv_file_flag
        csv_content = []
        if output_file_name is not None:
            csv_file_flag = True

        print_value('DLC info for: ' + os.path.abspath(input_file_name), csv_content)

        headers = ["Id", "Name", "Type", "Inputs", "Outputs", "Out Dims", "Runtimes", "Parameters"]
        col_sizes = [1+len(header) for header in headers]

        for row in self.rows:
            if row.param_count > 0:
                row.add_parm( "param count", get_si_notation(row.param_count, self.total_params))

            if row.macs > 0:
                row.add_parm( "MACs per inference", get_si_notation(row.macs, self.total_macs))

            if row.op_affinity != ir_graph.IrOpAffinity.UNSET:
                row.add_parm( "Op Affinity", row.layer_affinity)
            if show_memory_usage:
                if row.get_memory_cpu() > 0:
                    row.add_parm( "Memory needed", get_binary_si_notation(row.get_memory_cpu(), self.total_memory_cpu))

        for row in self.rows:
            col_sizes[0] = max(col_sizes[0], 1+row.id_width())
            col_sizes[1] = max(col_sizes[1], 1+row.name_width())
            col_sizes[2] = max(col_sizes[2], 1+row.type_width())
            col_sizes[3] = max(col_sizes[3], 1+row.input_name_and_type_width())
            col_sizes[4] = max(col_sizes[4], 1+row.output_name_and_type_width())
            col_sizes[5] = max(col_sizes[5], 1+row.output_dims_width())
            col_sizes[6] = max(col_sizes[6], 1+row.runtimes_width())
            col_sizes[7] = max(col_sizes[7], 1+row.parms_width())

        (model_version, total_params_str, total_macs_str, converter_command, quantizer_command,
         converter_version, model_copyright) = self.get_meta_data(self.total_params, self.total_macs, input_file_name)
        memory_data = self.get_memory_data(self.total_memory_cpu, self.total_memory_fxp, self.total_params, show_memory_usage)

        print_value(model_version, csv_content)
        print_value(model_copyright, csv_content)
        total_size = 2+2*len(col_sizes)-1+sum(col_sizes)
        print_value('-'*total_size)
        print_row(headers, col_sizes, csv_content)
        print_value('-'*total_size)

        for row in self.rows:
            row.dump(col_sizes, csv_content)
        print_value('-'*total_size)
        print_value("Note: The supported runtimes column assumes a processor target of Snapdragon 855", csv_content)
        print_value("Key : A:AIP\n      D:DSP\n      G:GPU\n      C:CPU\n", csv_content)

        def print_input_info():
            # Gather encoding info for app_write tensor printing
            encoding_info_dict = {}
            for row in self.rows:
                row.merge_encoding_infos(encoding_info_dict)

            app_write_tensors = set()
            for op in self.graph.get_ops():
                for input_tensor in op.inputs():
                    if input_tensor.is_app_write_tensor():
                        app_write_tensors.add(input_tensor)

            header = ['Input Name', 'Dimensions', 'Type']

            input_table = []
            have_encoding_info = False
            for aw_tensor in app_write_tensors:
                row = [aw_tensor.name(), ','.join([str(x) for x in aw_tensor.dims()]), aw_tensor.data_type_string()]
                # Append a new column if we have encoding info for this tensor
                if aw_tensor.name() in encoding_info_dict:
                    row.append(encoding_info_dict[aw_tensor.name()])
                    have_encoding_info = True
                input_table.append(row)

            # If any of the inputs have encoding info, we must append another column for each row
            if have_encoding_info:
                header.append('Encoding Info')
                for row in input_table:
                    if len(row) < len(header):
                        row.append('')
            input_table.insert(0, header)

            # Calculate max width in columns
            widths = [1 for _ in header]
            for i in range(len(input_table)):
                for j in range(len(widths)):
                    widths[j] = max(widths[j], 1+len(input_table[i][j]))

            # Print network input summary table
            hor_div = '-'*(sum(widths) + 2*len(widths) + 1)
            print_value(hor_div)
            for i in range(len(input_table)):
                print_row(input_table[i], widths, csv_content)
                if i == 0:
                    print_value(hor_div)
            print_value(hor_div + '\n')

        print_input_info()

        # Add unique op types to Meta data printing for dlc-info
        op_types = "Ops used by DLC: {}".format(", ".join(sorted(self.total_op_types)))
        print_value(total_params_str + '\n' + total_macs_str + '\n' + converter_command + '\n' + quantizer_command+ '\n' + converter_version +
                    "\n" + op_types, csv_content)

        print_value('Est. Steady-State Memory Needed to Run: ' + '\n'.join(memory_data), csv_content)

        print_value(('-' * total_size) + '\n')

        # Print cached blob information
        try:
            backend_cache_records = self.get_backend_cache_records()
            if backend_cache_records:
                self.dump_cache_info(backend_cache_records, csv_content)
        except Exception as e:
            raise Exception("Error Querying for Backend Cache Records in model:", self.model_filename, e)

        if output_file_name is not None:
            try:
                with open(output_file_name, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    for d in csv_content:
                        writer.writerow(d)

            except IOError:
                print("IOError: Cannot open CSV file " + output_file_name, file=sys.stderr)
                sys.exit(-1)

    def dump_runtime_info(self):
        count = 1
        for row in self.rows:
            print("Op %2d\t" % count + str(row.valid_cpu) + " | " + str(row.valid_gpu) + " | " + str(row.valid_dsp) + " | " + str(row.valid_aip))
            print("-" * 60)
            count = count + 1

    def dump_cache_info(self, cache_records, csv_content):
        """
            Prints Aix table with each record and its corresponding meta information
        @param aix_records: dictionary of the aix records and their meta info
        """

        headers = ["Cache Record Name", "snpe_version", "record_version", "backend_record_search_key",
                   "backend_type", "record_size", "Subnets"]
        col_sizes = [1 + len(header) for header in headers]
        max_col_size = 25
        for s in range(0, len(col_sizes)):
            col_sizes[s] = max(col_sizes[s], max_col_size)

        subnet_col_max_size = 10
        total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
        print_value("\nCache Info:", csv_content)
        row_to_print = []

        for cache_record_name, cache_meta_info in cache_records.items():

            print_list = [] # adding the rows that we want to print

            cache_meta_list = [cache_record_name]  # add the record name column

            # add everything after name but before Subnets(since Subnets have further info)
            for i in range(1, len(headers) - 1):
                cache_meta_list.append(cache_meta_info[headers[i]])

            subnet_col = "num_of_subnets: " + str(cache_meta_info['num_of_subnets'])
            cache_meta_list.append(subnet_col)
            print_list.append([cache_meta_list, csv_content])

            # Position of the 'Subnets' column in the table
            subnet_col_position = 7
            prep_subnet_info_row = lambda x : ["" for i in range(0, subnet_col_position-1)] + [x]
            # Add subnets meta info for record
            if cache_meta_info['compatibility']:
                for j in range(0, cache_meta_info['num_of_subnets']):
                    subnet_name = "subnet_" + str(j)
                    print_list.append([prep_subnet_info_row(subnet_name + ":"), csv_content])

                    # note: separated if cases for start/end ids so that they get printed one after the other for
                    #        better visual. Python was ordering them randomly even if OrderedDict was used.
                    if "start_layer_Id" in cache_meta_info[subnet_name].keys():
                        subnet_col = "start_layer_Id: " + str(cache_meta_info[subnet_name]["start_layer_Id"])
                        print_list.append([prep_subnet_info_row("  " + subnet_col), csv_content])
                        cache_meta_info[subnet_name].pop("start_layer_Id")

                    if "end_layer_Id" in cache_meta_info[subnet_name].keys():
                        subnet_col = "end_layer_Id: " + str(cache_meta_info[subnet_name]["end_layer_Id"])
                        print_list.append([prep_subnet_info_row("  " + subnet_col), csv_content])
                        cache_meta_info[subnet_name].pop("end_layer_Id")

                    for subnet_key, subnet_value in cache_meta_info[subnet_name].items():
                        subnet_col = subnet_key + ": "
                        if isinstance(subnet_value, list):
                            print_list.append([prep_subnet_info_row("  " + subnet_col), csv_content])
                            for value in subnet_value:
                                subnet_col_max_size = max(subnet_col_max_size, len(str(value)) + len(subnet_key) + 5)
                                value = "    " + value  # indent for visual
                                print_list.append([prep_subnet_info_row(value), csv_content])
                        else:
                            subnet_col += str(subnet_value)
                            print_list.append([prep_subnet_info_row("  " + subnet_col), csv_content])

                    col_sizes[-1] = subnet_col_max_size  # to account for long buffer names
                    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)

            else:
                # print warning if record is not compatible with current version of snpe
                err_msg = ' Warning: ' + cache_record_name + ' is incompatible with the latest version of SNPE.'
                remaining_col_len = total_size - len(err_msg) - 2  # to find where to put the trailing '|' for row
                # print error for record indented and in red color
                print_value('|' + "\033[91m{}\033[00m" .format(err_msg) + (" " * remaining_col_len) + '|', csv_content)

            row_to_print.append([print_list, col_sizes])

        print_headers(headers, col_sizes, csv_content, total_size)
        for i in range(len(row_to_print)):
            print_all(row_to_print[i][0], row_to_print[i][1])

    def dump_aix_info(self, aix_records, csv_content):
        """
            Prints Aix table with each record and its corresponding meta information
        @param aix_records: dictionary of the aix records and their meta info
        """

        print_value("\nAIP Info:", csv_content)
        headers = ["AIP Record Name", "nnc_version", "record_version", "hta_blob_id", "record_size", "Subnets"]
        col_sizes = [1 + len(header) for header in headers]

        max_col_size = 25
        for s in range(0, len(col_sizes)):
            col_sizes[s] = max(col_sizes[s], max_col_size)

        subnet_col_max_size = 40
        col_sizes[-1] = max(col_sizes[-1], subnet_col_max_size)  # to account for long buffer names

        total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
        print_value('-'*total_size)
        print_row(headers, col_sizes, csv_content)
        print_value('-'*total_size)
        for aix_record_name, aix_meta_info in aix_records.items():
            aix_meta_list = [aix_record_name]  # add the record name column

            # add everything after name but before Subnets(since Subnets have further info)
            for i in range(1, len(headers) - 1):
                aix_meta_list.append(aix_meta_info[headers[i]])

            subnet_col = "num_of_subnets: " + str(aix_meta_info['num_of_subnets'])
            aix_meta_list.append(subnet_col)
            print_row(aix_meta_list, col_sizes, csv_content)

            # Add subnets meta info for record
            if aix_meta_info['compatibility']:
                for j in range(0, aix_meta_info['num_of_subnets']):
                    subnet_name = "subnet_" + str(j)
                    print_row(["", "", "", "", "", subnet_name + ":"], col_sizes, csv_content)
                    # note: separated if cases for start/end ids so that they get printed one after the other for
                    #        better visual. Python was ordering them randomly even if OrderedDict was used.
                    if "start_layer_Id" in aix_meta_info[subnet_name].keys():
                        subnet_col = "start_layer_Id: " + str(aix_meta_info[subnet_name]["start_layer_Id"])
                        print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                        aix_meta_info[subnet_name].pop("start_layer_Id")
                    if "end_layer_Id" in aix_meta_info[subnet_name].keys():
                        subnet_col = "end_layer_Id: " + str(aix_meta_info[subnet_name]["end_layer_Id"])
                        print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                        aix_meta_info[subnet_name].pop("end_layer_Id")
                    for subnet_key, subnet_value in aix_meta_info[subnet_name].items():
                        subnet_col = subnet_key + ": "
                        if isinstance(subnet_value, list):
                            print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
                            for value in subnet_value:
                                value = "    " + value  # indent for visual
                                print_row(["", "", "", "", "", value], col_sizes, csv_content)
                        else:
                            subnet_col += str(subnet_value)
                            print_row(["", "", "", "", "", "  " + subnet_col], col_sizes, csv_content)
            else:
                # print warning if record is not compatible with current version of snpe
                err_msg = '    Warning: This record(' + aix_record_name + ') is incompatible with the latest version of SNPE.'
                remaining_col_len = total_size - len(err_msg) - 2  # to find where to put the trailing '|' for row
                # print error for record indented and in red color
                print_value('|' + "\033[91m{}\033[00m" .format(err_msg) + (" " * remaining_col_len) + '|', csv_content)

    def read_converter_command(self, dlc_file_path):
        archive = zipfile.ZipFile(dlc_file_path, 'r')
        if 'dlc.metadata' in archive.namelist():
            meta = archive.read('dlc.metadata').decode()
            for k, v in [line.split('=', 1) for line in meta.split('\n') if len(line) > 0]:
                if k == 'converter-command':
                    return v
        return 'N/A'

    def read_quantizer_command(self, dlc_file_path):
        archive = zipfile.ZipFile(dlc_file_path, 'r')
        if 'dlc.metadata' in archive.namelist():
            meta = archive.read('dlc.metadata').decode()
            for k, v in [line.split('=', 1) for line in meta.split('\n') if len(line) > 0]:
                if k == 'quantizer-command':
                    return v
        return 'N/A'

    def is_aix_record_present(self):
        archive = zipfile.ZipFile(self.model_filename, 'r')
        archive_filelist = archive.filelist
        for fileinfo in archive_filelist:
            if "aip" in fileinfo.filename or "hta" in fileinfo.filename or "aix" in fileinfo.filename:
                return True
        return False

    def get_aix_records(self):
        # TODO: There is currently no way to get aix records in dlcv4
        return self.model.get_aix_records()

    def get_backend_cache_records(self):
        self.cache_reader.populate_cache_records_from_dlc(self.model_filename)

        cache_records = {}
        for record_idx in range(0, self.cache_reader.record_count):
            record = self.cache_reader.get_record(record_idx)

            if record.is_valid:
               record_dict = {
                   'snpe_version': record.snpe_version,
                   'backend_type': record.backend_type,
                   'backend_record_search_key': record.backend_record_search_key,
                   'htp_optimization_level': record.htp_optimization_level,
                   'record_version': record.record_version,
                   'record_size': record.record_size,
                   'num_of_subnets': record.num_of_subnets,
                   'compatibility': record.is_compatible
               }

            for subnet_idx in range(0, record.num_of_subnets):
                subnet = record.get_subnet_at(subnet_idx)

                subnet_dict = {
                    'start_layer_Id': subnet.start_op_id,
                    'end_layer_Id': subnet.end_op_id -1,
                    'input_buffers': subnet.input_buffer_info,
                    'output_buffers': subnet.output_buffer_info
                }

                record_dict['subnet_' + str(subnet_idx)] = subnet_dict

            cache_records[record.record_name] = record_dict
        return cache_records

    def get_op_mapping(self):

        if self.op_mapping == {}:
            graph = self.model_reader.get_ir_graph()
            for idx,op in enumerate(graph.get_ops()):
                row  = OpRow(op, self.rows)
                self.op_mapping[idx] = row.name

        return self.op_mapping

    def get_meta_data(self, total_params, total_macs, input_file_name):

        model_version = 'Model Version: %s' %self.model_reader.custom_model_version
        total_params = 'Total parameters: %d (%d MB assuming single precision float)' %(total_params, total_params*4/(1024*1024))
        total_macs = 'Total MACs per inference: %s' %get_si_notation(total_macs, total_macs)
        converter_command = 'Converter command: {}'.format(self.model_reader.converter_command)
        quantizer_command = 'Quantizer command: {}'.format(self.model_reader.quantizer_command)
        converter_version = 'DLC created with converter version: {}'.format(self.model_reader.converter_version)
        model_copyright = 'Model Copyright:{}'.format(self.model_reader.copyright)

        return model_version, total_params, total_macs, converter_command, quantizer_command, converter_version, model_copyright

    def get_memory_data(self, total_memory_cpu, total_memory_fxp, total_params, show_full_usage):

        # sizes of fixed allocations in SNPE, measured with Massif
        symphony_fixed_allocs = 23068672
        hogl_fixed_allocs = 67141632 + 3145728

        mem_linux_cpu = total_memory_cpu + hogl_fixed_allocs + symphony_fixed_allocs
        mem_linux_fxp = total_memory_fxp + hogl_fixed_allocs + symphony_fixed_allocs
        mem_android_cpu = total_memory_cpu + symphony_fixed_allocs

        lines = []

        if show_full_usage:

            # print full usage data
            lines.append('') # start on new line
            lines.append('- On Linux CPU: %s' %get_binary_si_notation(mem_linux_cpu, None))
            lines.append('- On Linux CPU Quantized to 8-bit: %s' %get_binary_si_notation(mem_linux_fxp, None))
            lines.append('- On Android CPU: %s' %get_binary_si_notation(mem_android_cpu, None))
        else:

            # print abridged usage data
            lines.append('%s' %get_binary_si_notation(total_memory_cpu, None))

        return lines

    def get_input_dims(self):
        row = self.rows[0]
        return row.outputs_string(0)

    def get_total_macs(self):
        return self.total_macs

    def get_total_params(self):
        return self.total_params

    def types_info(self):
        name_and_type = {}
        for row in self.rows:
            name_and_type.update({row.name:row.type})
        return name_and_type

    def types_info_by_id(self):
        id_and_type = {}
        for row in self.rows:
            id_and_type.update({row.id: row.type})
        return id_and_type

    def op_names_by_id(self):
        id_and_type = {}
        for id,row in enumerate(self.rows):
            id_and_type.update({id: row.name})
        return id_and_type

    def ids_layer(self):
        name_and_id = {}
        for id,row in enumerate(self.rows):
            name_and_id.update({row.name: id})
        return name_and_id

    def params_info(self):
        name_and_parm = OrderedDict()
        for row in self.rows:
            m = max(len(row.get_parm_list()), len(row.get_input_list()))
            m = max(m,len(row.get_output_list()))
            parms = []
            for i in range(0,m-1):
                parms.append(row.get_parm(i))
            name_and_parm.update({row.name:parms})
        return name_and_parm

    def dims_info(self):
        name_and_dims = OrderedDict()
        for row in self.rows:
            output_dims = ['x'.join(map(str, dim)) for dim in row.output_dims_list]
            name_and_dims.update({row.name: output_dims})
        return name_and_dims

    def weights_info(self):
        name_and_weights = OrderedDict()
        for row in self.rows:
            name_and_weights.update({row.name: row.op.get_weights()})

        return name_and_weights

    def get_output_names(self):
        output_names = OrderedDict()
        for row in self.rows:
            output_names.update({row.name: row.get_output_list()})
        return output_names

    def is_aix_enabled(self):
        # TODO: we need a way to read aix records before we can say it's enabled
        return False
        # return self.model_reader.is_aix_enabled()

    def get_model_copyright(self):
        return self.model_reader.copyright

