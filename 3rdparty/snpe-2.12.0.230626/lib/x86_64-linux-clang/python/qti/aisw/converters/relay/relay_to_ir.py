# ==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import re
import sys

# tvm, relay
try:
    import tvm
    from tvm import relay
except ModuleNotFoundError as e:
    print("Error while importing Relay...\n")
    raise e
except ImportError as e:
    print("TVM not found in PYTHONPATH. Ensure PYTHONPATH includes <path/to/tvm>/python.\n"
          "You can download and install TVM from https://tvm.apache.org/docs/install/from_source.html\n")
    sys.exit(1)
except Exception as e:
    print("Error while importing TVM...\n")
    raise e

from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ConstantOp,
    Op,
)
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.utils.converter_utils import (
    converter_type,
    log_assert,
    log_debug1,
    log_debug2,
    log_debug3,
    log_error,
    log_verbose,
    log_warning,
)

from qti.aisw.converters.relay.importers.relay_importer import RelayImporter
from qti.aisw.converters.relay.utils import get_key_from_expr
from .translations import RelayTranslations

global QUIR_GRAPH
global RELAY_PARAMS
global EXPR_TO_SOURCE_INFO
global CONVERTER_CTX


def get_op_type(op_type):
    # Some op type names are repeated since they are stored under a class hierarchy. In order
    # to ensure the correct translations are used these op classes leverage the full op names
    SPECIAL_CLASSES = ['qnn']
    if not op_type.split(('.'))[0] in SPECIAL_CLASSES:
        op_type = str(op_type).split(('.'))[-1]
    log_debug2("op_type in get_translation {}", op_type)
    return converter_type(op_type, 'relay')


def get_translation(expr):
    op_type = get_op_type(str(expr.op.name))
    if op_type in RelayTranslations.translations:
        return RelayTranslations.translations[op_type]
    else:
        raise TypeError("Unsupported Op type {}".format(expr.op.name))


class RelayConverterContext:
    """
    Class that contains all data structures and methods for Op Conversion
    """
    def __init__(self, quir_graph: IROpGraph, expr_to_source_info_dict: dict=None):
        self.expr_to_source_info_dict = expr_to_source_info_dict
        if self.expr_to_source_info_dict:
            log_verbose("Output Names in expr_to_source_info Dict:")
            for expr, source_info in self.expr_to_source_info_dict.items():
                op_name = source_info.get_op_name()
                if op_name:
                    log_verbose("\t {}: {}", op_name, source_info.get_output_names())
                else:
                    log_verbose("\t{}", source_info.get_output_names())
            log_verbose("\n")
        self.type_count = {}
        self.quir_graph = quir_graph

    def get_op_name(self, expr: relay.expr, op_type: str, legacy_translation_key: str = None):
        """
        Generates Op name that is unique using ref count per Op Type
        :param expr: Relay Expr
        :param op_type: QuIR Op Type
        :param legacy_translation_key: Legacy Python IR op type
        :return: Str
        """
        key = get_key_from_expr(expr)
        op_name = self.expr_to_source_info_dict[key].get_op_name()
        if not op_name:
            count = self.type_count.get(op_type, 0)
            self.type_count[op_type] = count + 1
            if legacy_translation_key:
                name_prefix_str = str(legacy_translation_key)
            else:
                name_prefix_str = str(op_type)
            op_name = "%s_%d" % (name_prefix_str, count)
        self.expr_to_source_info_dict[key].set_op_name(op_name)
        log_verbose("op_name {}", op_name)
        return op_name

    def get_input_names(self, expr: relay.expr):
        """
        Get Input Names for input Relay Expr. It uses recursive tree traversal to get output names of
        inputs to the Input Expr
        :param expr: Relay Expr
        :return: List of input names
        """
        inputs = []
        if isinstance(expr, relay.Call):
            for arg in expr.args:
                outputs = self.get_output_names(arg)
                log_verbose("Call outputs {}", outputs)
                inputs.extend(outputs)
        elif isinstance(expr, relay.Var):
            k = get_key_from_expr(expr)
            output_names = self.expr_to_source_info_dict[expr].get_output_names()
            if output_names:
                log_verbose("Var name {} outputs {}", expr.name_hint, output_names)
                inputs.append(output_names)
            else:
                raise KeyError("Span or Expr for {} not found in dictionary expr_to_source_info_dict".format(expr))
        elif isinstance(expr, relay.TupleGetItem):
            log_verbose("tuple item input index {}", expr.index)
            tuple_inputs = self.get_output_names(expr.tuple_value)[expr.index]
            log_verbose("Appending tuple item input {}", tuple_inputs)
            inputs.extend(tuple_inputs)
        elif isinstance(expr, relay.Tuple):
            for elem in expr.fields:
                log_verbose("inputs before Tuple {}", inputs)
                inputs.extend(self.get_output_names(elem))
                log_verbose("inputs after Tuple {}", inputs)
        else:
            raise TypeError("Unsupported Expr type {} for get_input_names".format(type(expr)))

        return inputs

    def get_input_shapes(self, expr: relay.expr):
        """
        Get Buffer Shapes from QuIR Graph for inputs to the Relay expr
        :param expr: Relay Expr
        :return: List of input shapes
        """
        inputs = self.get_input_names(expr)
        input_shapes = []
        for input_name in inputs:
            if self.quir_graph.has_buffer(input_name):
                input_shape = self.quir_graph.get_buffer(input_name).shape
            elif input_name in RELAY_PARAMS:
                input_shape = RELAY_PARAMS[input_name].shape
                input_shape = list(map(int, input_shape))
            else:
                raise KeyError("input_name {} is not found in graph buffers, nor RELAY_PARAMS".format(input_name))
            input_shapes.append(input_shape)
        log_verbose("input_shapes {}", *zip(inputs, input_shapes))
        return input_shapes

    def get_output_names(self, expr: relay.expr, num_outputs: int=None):
        """
        Get output names of given Relay Expr
        :param expr: Relay Expr
        :param num_outputs:
        :return:
        """

        key = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[key].get_output_names()
        if not output_names:
            if isinstance(expr, relay.Var):
                log_verbose("Var name {}", expr.name_hint)
                output_names = [expr.name_hint]
                self.expr_to_source_info_dict[key].set_output_names(output_names)
            elif isinstance(expr, relay.Tuple):
                output_names = []
                for elem in expr.fields:
                    log_verbose("tuple outputs before {}", output_names)
                    output_names.extend(self.get_output_names(elem))
                    log_verbose("tuple outputs after {}", output_names)
            elif isinstance(expr, relay.TupleGetItem):
                output_names = [self.get_output_names(expr.tuple_value)[expr.index]]
                log_verbose("Appending tuple item output {}", output_names)
            else:
                # expr is not in self.expr_to_source_info_dict
                if num_outputs:
                    output_names = self.generate_output_names(expr, num_outputs)
                    self.expr_to_source_info_dict[key].set_output_names(output_names)
                else:
                    log_error("Unknown expr:\n{}\ntype {}\n", expr, type(expr))
                    raise KeyError("Unknown Expr found while getting output names")
        else:
            if num_outputs is not None:
                log_assert(len(output_names)==num_outputs, "output_names not match num_outputs for expr:\n{}", expr)

        return output_names

    def generate_output_names(self, expr: relay.expr, num_outputs: int):
        """
        Generate output tensor names for given Relay Expr since they were not already provided
        :param expr: Relay Expr
        :param num_outputs:
        :return:
        """
        k = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[k].get_output_names()
        if not output_names:
            output_names = [self.expr_to_source_info_dict[k].get_op_name() + '_' +
                            str(i) for i in range(num_outputs)]
            log_verbose("generated output names {}", output_names)
        return output_names

    def add_op_to_graph(self,
                        expr: relay.expr,
                        op: Op,
                        input_names: list,
                        output_names: list,
                        axis_formats: list=None,
                        idx: int=-1):
        """
        Add QuIR Op to QuIR OpGraph and update the dictionary of expr to output names
        :param expr: Relay Expr
        :param op: QuIR Op
        :param input_names: List of input names
        :param output_names: List of output names
        :param axis_formats:
        :param idx: Index in graph to insert the Node
        :return: QuIR OpNode
        """
        key = get_key_from_expr(expr)
        output_names = self.expr_to_source_info_dict[key].get_output_names()
        if not output_names:
            self.expr_to_source_info_dict[key].set_output_names(output_names)

        # add Constant Op for input_name in relay_param but not in the graph.
        for input_name in input_names:
            if not QUIR_GRAPH.has_buffer(input_name):
                log_assert(input_name in RELAY_PARAMS,
                           "Input {} not found in Graph or Params", input_name)
                log_debug3("Adding ConstantOp for {} due to op {}", input_name, op.name)

                const_input_tensor = RELAY_PARAMS[input_name]
                if isinstance(const_input_tensor, (tvm.runtime.ndarray.NDArray, tvm.runtime.NDArray)):
                    const_input_tensor = const_input_tensor.asnumpy()
                QUIR_GRAPH.add(ConstantOp(input_name, const_input_tensor),
                               input_names=[],
                               output_names=[input_name])

        return QUIR_GRAPH.add(op, input_names, output_names, axis_formats, idx)


class RelayConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(RelayConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--dump_relay', type=str, default=None,
                                       help="Dump Relay ASM and Params at the path provided with the argument\n"
                                            "Usage: --dump_relay <path_to_dump>")

    def __init__(self, args, importer: RelayImporter=None, mod: tvm.IRModule=None, params: dict=None, **kwargs):
        super(RelayConverterFrontend, self).__init__(args,
                                                     **kwargs)
        self.importer = importer
        if self.importer and isinstance(self.importer, RelayImporter):
            self.relay_mod, self.relay_params, self.expr_to_source_info_dict = self.importer.convert_to_relay(self.input_model_path)
        else:
            mod = mod
            params = params
            if not mod or not params:
                raise SyntaxError("{} should be initialized with either an importer or with (mod, params). "
                                  "None of these provided.".format(self.__class__.__name__))
            self.expr_to_source_info_dict = {}
            self.relay_mod = mod
            self.relay_params = params

        if args.dump_relay:
            self.dump_relay_data(args)
        self.converter_context = RelayConverterContext(self.graph,
                                                       expr_to_source_info_dict=self.expr_to_source_info_dict)
        self._init_globals()

    def dump_relay_data(self, args):
        ########## debugging ###########
        import os
        if not args.dump_relay:
            path = '/'.join(os.path.realpath(args.input_network).split('/')[:-1])
        else:
            path = args.dump_relay

        log_verbose("Dumping Relay data at {}", path)

        full_mod_txt_path = os.path.join(path, "mod.txt")
        full_mod_json_path = os.path.join(path, "mod.json")
        self.dump_mod(full_mod_txt_path, full_mod_json_path)

        full_params_path = os.path.join(path, "params.txt")
        self.dump_params(full_params_path)
        ########## end debugging ###########

    def _init_globals(self):
        global QUIR_GRAPH
        QUIR_GRAPH = self.graph

        global RELAY_PARAMS
        RELAY_PARAMS = self.relay_params

        global EXPR_TO_SOURCE_INFO
        EXPR_TO_SOURCE_INFO = self.expr_to_source_info_dict

        global CONVERTER_CTX
        CONVERTER_CTX = self.converter_context

    def dump_params(self, file_name):
        with open(file_name, "w") as f:
            for k, v in self.relay_params.items():
                f.write(k)
                f.write(':')
                f.write(str(v))
                f.write('\n')

    def dump_mod(self, mod_txt_path, mod_json_path):
        with open(mod_txt_path, "w") as f:
            f.write(self.relay_mod.astext(show_meta_data=False))

        with open(mod_json_path, "w") as f:
            f.write(tvm.ir.save_json(self.relay_mod))

    @staticmethod
    def add_input(expr: relay.expr):
        if not isinstance(expr, relay.Var):
            return

        global QUIR_GRAPH
        global RELAY_PARAMS
        global EXPR_TO_SOURCE_INFO
        global CONVERTER_CTX

        var_name = str(expr).split("\n")[1].lstrip("%v")

        k = get_key_from_expr(expr)

        if var_name in RELAY_PARAMS:
            output_names = EXPR_TO_SOURCE_INFO[k].get_output_names()
            if not output_names:
                EXPR_TO_SOURCE_INFO[k].set_output_names([var_name])
            param = RELAY_PARAMS[var_name]
            log_verbose("param {}", var_name)
            log_verbose("shape {}", param.shape)
        else:
            log_verbose("input {}", var_name)
            log_verbose("type {}", type(expr))
            log_verbose('shape {}', list(expr.type_annotation.shape))
            input_shape = [int(val) for val in expr.type_annotation.shape]
            input_node = QUIR_GRAPH.add_input(var_name, input_shape)
            output_names = EXPR_TO_SOURCE_INFO[k].get_output_names()
            if not output_names:
                EXPR_TO_SOURCE_INFO[k].set_output_names([var_name])

            # populate quantization info for input var
            key = get_key_from_expr(expr)
            encodings = EXPR_TO_SOURCE_INFO[key].get_encodings()
            if encodings:
                QUIR_GRAPH.set_overridden_encoding(input_node.op.name, encodings, is_param=False)

    @staticmethod
    def add_constant(expr: relay.expr):
        if not isinstance(expr, relay.Constant):
            return

        global QUIR_GRAPH
        global RELAY_PARAMS
        global EXPR_TO_SOURCE_INFO
        global CONVERTER_CTX

        key = get_key_from_expr(expr)
        output_names = EXPR_TO_SOURCE_INFO[key].get_output_names()
        if not output_names:
            constant_name = CONVERTER_CTX.get_op_name(expr, 'relay_constant', "")
            EXPR_TO_SOURCE_INFO[key].set_output_names([constant_name])
        else:
            constant_name = output_names[0]
        # update relay_params
        constant_array = expr.data
        RELAY_PARAMS[constant_name] = constant_array

    @staticmethod
    def add_op(expr: relay.expr):
        if isinstance(expr, relay.Call):
            # op_name = str(expr.op.name).replace("nn.", "")
            # log_verbose("name {}", expr.op.name)
            log_debug1("")
            log_debug1("Relay Op name {}", expr.op)

            ##### DEBUG PRINTS #####
            attributes = {}
            if expr.attrs:
                for attr in expr.attrs.list_field_info():
                    attributes[attr.name] = {}
                    attributes[attr.name]['value'] = getattr(expr.attrs, attr.name)
            log_verbose("attributes:")
            for k, v in attributes.items():
                log_verbose("\t{}:{}", k, v)
            ##### END DEBUG #####

            translation = get_translation(expr)

            global CONVERTER_CTX
            global RELAY_PARAMS
            translation.add_op(expr, QUIR_GRAPH, converter_context=CONVERTER_CTX, relay_params=RELAY_PARAMS)
        else:
            pass

    @staticmethod
    def visit_module(expr: relay.expr):
        log_debug2("")
        log_debug2("##### NEW OP Translation #####")
        if isinstance(expr, relay.Var):
            RelayConverterFrontend.add_input(expr)
        elif isinstance(expr, relay.Constant):
            RelayConverterFrontend.add_constant(expr)
        elif isinstance(expr, relay.Call):
            RelayConverterFrontend.add_op(expr)
        else:
            log_verbose("{}", type(expr))

        log_debug2("\n")

    def convert_to_ir(self):
        relay.analysis.post_order_visit(self.relay_mod["main"], RelayConverterFrontend.visit_module)
        self.graph.eval_macs_params()
        return self.graph

    def convert(self):
        # Wrapper for combination of convert_to_relay and convert_to_ir
        return self.convert_to_ir()
