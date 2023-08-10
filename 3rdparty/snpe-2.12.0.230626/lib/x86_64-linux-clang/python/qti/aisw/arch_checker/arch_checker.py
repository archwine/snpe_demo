# =============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
import pandas as pd
from collections import OrderedDict
from abc import abstractmethod
import numpy as np

from qti.aisw.converters.common import ir_graph

import qti.aisw.arch_checker.constants as const

class ArchChecker:
    def __init__(self, c_ir_graph, constraints_json, output_file, logger):
        self.c_ir_graph = c_ir_graph
        self.constraints_json = constraints_json
        self.output_file = output_file
        self.logger = logger

    def _get_in_out_info(self, all_tensors):
        ret = []
        for tensor in all_tensors:
            cur = tensor.name() + ":" + str(tensor.dims())
            ret.append(cur)
        return ','.join(ret)

    def _get_params(self, op):
        ret = {}
        for name in op.attrs.list_names():
            # parse scalar attrs
            if op.attrs.get_attr_type(name) == ir_graph.QNN_PARAMTYPE_SCALAR:
                if op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_8 or \
                   op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_SFIXED_POINT_8:
                    ret[name] = op.attrs.get_int8(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_16 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_SFIXED_POINT_16:
                    ret[name] = op.attrs.get_int16(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_32 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_SFIXED_POINT_32:
                    ret[name] = op.attrs.get_int32(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_INT_64:
                    ret[name] = op.attrs.get_int64(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_8 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UFIXED_POINT_8:
                    ret[name] = op.attrs.get_uint8(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_16 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UFIXED_POINT_16:
                    ret[name] = op.attrs.get_uint16(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_32 or \
                     op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UFIXED_POINT_32:
                    ret[name] = op.attrs.get_uint32(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UINT_64:
                    ret[name] = op.attrs.get_uint64(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_FLOAT_16 or \
                    op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_FLOAT_32:
                    ret[name] = op.attrs.get_float(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_BOOL_8:
                    ret[name] = op.attrs.get_bool(name)
                elif op.attrs.get_data_type(name) == ir_graph.QNN_DATATYPE_UNDEFINED:
                    ret[name] = op.attrs.get_string(name)
            # parse tensor attributes
            elif op.attrs.get_attr_type(name) == ir_graph.QNN_PARAMTYPE_TENSOR:
                tensor_data = op.attrs.get_static_tensor_data(name)
                # skip tensor params that has large data (tensor size > 10)
                if tensor_data.size > 10:
                    continue
                ret[name] = tensor_data.tolist()
            else:
                self.logger.error("Unknown param type")
                exit(-1)
        return ret

    def _get_input_ops(self, op):
        ret = []
        in_tensors = [i for i in op.inputs()]
        for t in in_tensors:
            producer = t.get_producer()
            if producer:
                ret.append(producer.name)
        return ret

    def _get_output_ops(self, op):
        ret = []
        out_tensors = [i for i in op.outputs()]
        for t in out_tensors:
            consumers = [c.name for c in t.get_consumers()]
            ret.extend(consumers)
        return ret

    def run_checks(self):
        with open(self.constraints_json) as f:
            constraints = json.load(f)
        df = pd.DataFrame(columns=const.OUTPUT_CSV_HEADER)
        # graph check
        for constraint_id in constraints["graph"]:
            check = constraints["graph"][constraint_id]
            if eval(check["condition"]):
                tmp_df = dict.fromkeys(const.OUTPUT_CSV_HEADER, 'N/A')
                tmp_df[const.O_C_GRAPH_NODENAME] = "Graph"
                tmp_df[const.O_C_ISSUE] = eval(check["issue"])
                tmp_df[const.O_C_RECOMM] = eval(check["recomm"])
                df = df.append(tmp_df, ignore_index=True)
        # single node check
        for op in self.c_ir_graph.get_ops():
            for rule_id in constraints["single_node"]:
                check = constraints["single_node"][rule_id]
                if eval(check["condition"]):
                    tmp_df = dict.fromkeys(const.OUTPUT_CSV_HEADER, 'N/A')
                    tmp_df[const.O_C_GRAPH_NODENAME] = op.name
                    tmp_df[const.O_C_TYPE] = op.type
                    tmp_df[const.O_C_INPUTS] = self._get_in_out_info(op.inputs())
                    tmp_df[const.O_C_OUTPUTS] = self._get_in_out_info(op.outputs())
                    tmp_df[const.O_C_ISSUE] = eval(check["issue"])
                    tmp_df[const.O_C_RECOMM] = eval(check["recomm"])
                    tmp_df[const.O_C_PARAM] = self._get_params(op)
                    tmp_df[const.O_C_PRODUCER] = self._get_input_ops(op)
                    tmp_df[const.O_C_CONSUMERS] = self._get_output_ops(op)
                    df = df.append(tmp_df, ignore_index=True)
        # patterns check
        for rule_id in constraints["patterns"]:
            check = constraints["patterns"][rule_id]
            ops = eval(check["condition"])
            for op in ops:
                tmp_df = dict.fromkeys(const.OUTPUT_CSV_HEADER, 'N/A')
                tmp_df[const.O_C_GRAPH_NODENAME] = op.name
                tmp_df[const.O_C_TYPE] = op.type
                tmp_df[const.O_C_INPUTS] = self._get_in_out_info(op.inputs())
                tmp_df[const.O_C_OUTPUTS] = self._get_in_out_info(op.outputs())
                tmp_df[const.O_C_ISSUE] = eval(check["issue"])
                if isinstance(ops, dict) and ops[op]:
                    tmp_df[const.O_C_RECOMM] = eval(check["recomm_alt"])
                else:
                    tmp_df[const.O_C_RECOMM] = eval(check["recomm"])
                tmp_df[const.O_C_PARAM] = self._get_params(op)
                tmp_df[const.O_C_PRODUCER] = self._get_input_ops(op)
                tmp_df[const.O_C_CONSUMERS] = self._get_output_ops(op)
                df = df.append(tmp_df, ignore_index=True)
        df.to_csv(self.output_file, index=False)

    @abstractmethod
    def is_8bit(self):
        """Return True if 8 bit"""
        pass

    def is_conv(self, op):
        if op.type == ir_graph.QNN_OP_CONV_2D:
            return True
        elif op.type == ir_graph.QNN_OP_DEPTH_WISE_CONV_2D:
            return True
        elif op.type == ir_graph.QNN_OP_TRANSPOSE_CONV_2D:
            return True
        return False

    def is_conv_channel_mul_of(self, op, mult):
        if op.get_input_shapes()[0][const.INDEX_CHANNEL] % mult != 0:
            return False
        elif op.get_output_shapes()[0][const.INDEX_CHANNEL] % mult != 0:
            return False
        return True

    def is_conv_channel_less_than(self, op, channel_size):
        if op.get_input_shapes()[0][const.INDEX_CHANNEL] < channel_size:
            return True
        elif op.get_output_shapes()[0][const.INDEX_CHANNEL] < channel_size:
            return True
        return False

    def is_start_conv_seq(self, op):
        parent = op.inputs()[0].get_producer()
        if parent:
            if self.is_conv(parent):
                return False
            if self.is_activation(parent.type):
                grandparent = parent.inputs()[0].get_producer()
                if grandparent and self.is_conv(grandparent):
                    return False
        return True

    def is_conv_padding_in_middle(self):
        op_list = self.c_ir_graph.get_ops()
        bad_ops = []
        for op in op_list:
            if self.is_conv(op):
                padding = list(self._get_params(op)['pad_amount'])
                if np.sum(padding) != 0:
                    if not self.is_start_conv_seq(op):
                        bad_ops.append(op)
        return bad_ops

    def is_conv_seq_low_channel(self, channel_size):
        op_list = self.c_ir_graph.get_ops()
        bad_ops = {}
        for op in op_list:
            if self.is_conv(op):
                if self.is_conv_channel_less_than(op, channel_size):
                    if not self.is_start_conv_seq(op):
                        bad_ops[op] = True
                    else:
                        bad_ops[op] = False
        return bad_ops

    def is_activation(self, op_type):
        act_list = [ir_graph.QNN_OP_RELU, ir_graph.QNN_OP_SIGMOID,
        ir_graph.QNN_OP_TANH, ir_graph.QNN_OP_HARD_SWISH, ir_graph.QNN_OP_RELU_MIN_MAX]
        if op_type in act_list:
            return True
        return False

    def get_sn5_recomm(self, op):
        recomm = 'Try not to use {} if possible.'.format(op.type)
        if op.type == ir_graph.QNN_OP_TRANSPOSE:
            recomm += ' If necessary, Reshape is better than Transpose.'

            perm = self._get_params(op)['perm']
            if perm == [0,3,1,2] or perm == [0,2,3,1]:
                recomm += ' This Transpose is {}, which is probably alright.'.format(perm)
        return recomm
