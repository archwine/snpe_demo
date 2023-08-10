# ==============================================================================
#
#  Copyright (c) 2021,2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import abstractmethod, ABCMeta
from qti.aisw.converters.relay.utils import get_key_from_expr
from qti.tvm.relay.expr_functor import ExprVisitor
import tvm
from collections import defaultdict


class RelaySourceInfo(object):

    def __init__(self):
        self.encodings = None
        self.op_name = None
        self.output_names = None

    def get_encodings(self):
        return self.encodings

    def set_encodings(self, encodings):
        self.encodings = encodings.copy()

    def get_op_name(self):
        return self.op_name

    def set_op_name(self, op_name):
        self.op_name = op_name

    def get_output_names(self):
        return self.output_names

    def set_output_names(self, output_names):
        self.output_names = output_names.copy()


class RelaySpanParser(ExprVisitor):

    def __init__(self, expr_to_source_info_dict):
        super().__init__()
        self.expr_to_source_info_dict = expr_to_source_info_dict
        self.op_name_count = {}

    def visit_call(self, call):
        super().visit_call(call)
        op_name, output_names = self.get_call_info(call)
        self.populate_source_info(call, op_name, output_names)

    def visit_constant(self, const):
        op_name, output_names = self.get_constant_info(const)
        self.populate_source_info(const, op_name, output_names)

    def visit_var(self, var):
        op_name, output_names = self.get_var_info(var)
        self.populate_source_info(var, op_name, output_names)

    def get_call_info(self, call):
        span = call.span
        if isinstance(call.span, tvm.relay.SequentialSpan):
            # for activation we use last one since the order of spans in SequentialSpan is topological order
            span = call.span.spans[-1]
        if span is None:
            return '', []
        op_name = span.source_name.name
        relay_op_type = str(call.op.name).split('.')[-1]
        op_name = self.get_op_name(op_name, relay_op_type)

        output_names = span.output_names
        # for layout transform, we use layout and output_name make op_name/output_name
        # since it is more readable in generated QNNIR
        if call.op.name == 'layout_transform':
            layout = call.attrs.dst_layout.lower()
            input_key = get_key_from_expr(call.args[0])
            input_names = self.expr_to_source_info_dict[input_key].get_output_names()
            if input_names:
                op_name = '{}.{}'.format(input_names[0], layout)
                output_names = [op_name]
        return op_name, output_names

    def get_constant_info(self, const):
        span = const.span
        if isinstance(const.span, tvm.relay.SequentialSpan):
            # for params we use first one since the order of spans in SequentialSpan is topological order
            span = const.span.spans[0]
        if span is None:
            return '', []
        output_names = span.output_names
        if len(output_names) == 0:
            # for some constant created in frontend IR to relay, the output_name
            # could be empty, in this case, set output_name to op_name
            # e.g.
            # Constant(span=LSTM|LSTM_0|) => op_name in QNNIR: LSTM_0.constant_0
            #                             => output_names[0] = LSTM_0.constant_0
            op_name = span.source_name.name
            op_name = self.get_op_name(op_name, 'constant')
            output_names.append(op_name)
        return output_names[0], output_names

    def get_var_info(self, var):
        op_name = var.name_hint
        return op_name, [op_name]

    def populate_source_info(self, expr, op_name, output_names):
        key = get_key_from_expr(expr)
        if op_name:
            self.expr_to_source_info_dict[key].set_op_name(op_name)

        # set output_names if all output_names are not empty
        if len(output_names) and all(output_names):
            self.expr_to_source_info_dict[key].set_output_names(output_names)

    def get_op_name(self, op_name, relay_op_type):
        # generate relay op name by concatenating op_name from frontend and relay op type,
        # this help user to know how this op are translated to small ops
        # e.g.,
        # nn.conv2d (span=Conv|Conv_1|) => op_name in QNNIR: Conv_1.conv2d_0
        # nn.bias_add (span=Conv|Conv_1|Conv_out) => op_name in QNNIR: Conv_1.bias_add_0
        # the op_name could be empty for some models, these op_names will be generated in get_op_name in RelayConverterContext later
        if op_name and relay_op_type:
            op_name = '{}.{}'.format(op_name, relay_op_type)
            count = self.op_name_count.get(op_name, 0)
            self.op_name_count[op_name] = count + 1
            op_name = '{}_{}'.format(op_name, count)
        return op_name


class RelayImporter(object):
    __metaclass__ = ABCMeta

    def __init__(self, args):
        self.mod = None
        self.params = None
        self.expr_to_source_info_dict = defaultdict(RelaySourceInfo)

    @abstractmethod
    def convert_to_relay(self, input_model_path, **kwargs):
        """
        :param input_model_path: String representing path to source model
        :param kwargs:
        :return: Relay Module, Relay Params, [Expr to Output Names Dict]
        """
        raise NotImplementedError("convert_to_relay not implemented for {}".format(str(self.__class__.__name__)))
