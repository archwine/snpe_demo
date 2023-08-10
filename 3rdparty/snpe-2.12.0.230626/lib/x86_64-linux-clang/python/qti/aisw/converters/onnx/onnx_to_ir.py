# ==============================================================================
#
#  Copyright (c) 2018-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import faulthandler
import multiprocessing
import queue
import signal
import sys
import traceback
from packaging import version
import yaml

from qti.aisw.converters.common.utils import code_to_message

try:
    import onnx
except ImportError as e:
    raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(str(e), str(sys.path)))

from qti.aisw.converters.common.converter_ir import op_policies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.converter_ir.op_graph import InputLayout
from qti.aisw.converters.common.converter_base import ConverterFrontend
from .util import *
from . import onnx_translations

class OnnxConverterContext(object):
    def __init__(self, graph):
        """
        This class contains information regarding the weights obtained from WeightProvider.
        Any Other information that needs to be propagated can be added here without changing
        the graph class.
        :type graph: IROpGraph
        :type weights: WeightProvider
        """

        self.ir_graph = graph
        self.weights = []
        self.tensor_to_np_dtype = {}
        # TODO: deprecate it after 0d tensor is fully supported
        self.scalar_tensor = set()

# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(OnnxConverterFrontend.ArgParser, self).__init__(**kwargs)
            # add command-line options custom to onnx converter
            self.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                       help='Evaluates the model without actually converting any ops, and '
                                            'returns unsupported ops/attributes as well as unused inputs and/or '
                                            'outputs if any. Leave empty or specify "info" to see dry run as a '
                                            'table, or specify "debug" to show more detailed messages only"')
            self.add_optional_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The name and dimension of all the input buffers to the network specified in\n"
                                            "the format [input_name comma-separated-dimensions],\n"
                                            "for example: 'data' 1,224,224,3. \n"
                                            "Note that the quotes should always be included in order to handle special\n"
                                            "characters, spaces, etc.\n"
                                            "NOTE: This feature works only with Onnx 1.6.0 and above")
            self.add_optional_argument('-n', '--no_simplification', action='store_true', default=False,
                                       help="Do not attempt to simplify the model automatically. This may prevent some models from properly converting \n"
                                            "when sequences of unsupported static operations are present.")
            self.add_optional_argument('-b', '--batch', type=int, default=None,
                                       help="The batch dimension override. This will take the first dimension of all "
                                            "inputs and treat it as a batch dim, overriding it with the value provided "
                                            "here. For example:\n"
                                            "--batch 6\n"
                                            "will result in a shape change from [1,3,224,224] to [6,3,224,224].\n"
                                            "If there are inputs without batch dim this should not be used and each input "
                                            "should be overridden independently using -d option for input dimension overrides.")
            self.add_optional_argument('-s', '--define_symbol', nargs=2, action='append',
                                       metavar=('SYMBOL_NAME', 'VALUE'),
                                       help="This option allows overriding specific input dimension symbols. For instance you "
                                            "might see input shapes specified with variables such as :\n"
                                            "data: [1,3,height,width]\n"
                                            "To override these simply pass the option as:\n"
                                            "--define_symbol height 224 --define_symbol width 448\n"
                                            "which results in dimensions that look like:\n"
                                            "data: [1,3,224,448]")
            self.add_optional_argument('--dump_inferred_model', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--dump_value_info', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--dump_custom_io_config_template', type=str, default="",
                                 help='Dumps the yaml template for Custom I/O configuration. This file can'
                                 'be edited as per the custom requirements and passed using the option --custom_io'
                                 'Use this option to specify a yaml file to which the custom IO config template is dumped.')

    def __init__(self, args, *, custom_op_factory=None):
        super(OnnxConverterFrontend, self).__init__(args,
                                                    naming_policy=OnnxNamePolicy(),
                                                    shape_inference_policy=OnnxShapeInferencePolicy(),
                                                    axis_order=AxisOrders.ONNX,
                                                    custom_op_factory=custom_op_factory)
        self.translations = onnx_translations.OnnxTranslations
        self.dry_run = args.dry_run
        self.no_simplification = args.no_simplification
        self.dump_inferred_model = args.dump_inferred_model
        self.dump_value_info = args.dump_value_info
        self.op_info = onnx_translations.OpVersionInfo()
        self.converter_op_package_lib = args.converter_op_package_lib
        self.dump_custom_io_config_template = args.dump_custom_io_config_template
        if args.input_dim is not None:
            (in_names, in_dims) = list(zip(*args.input_dim))
            self.input_names = in_names
            self.input_dims = in_dims
        else:
            self.input_names = None
            self.input_dims = None

        self.define_symbols = None
        if args.define_symbol is not None:
            self.define_symbols = {item[0]: item[1] for item in args.define_symbol}

        self.batch = None
        if args.batch is not None:
            self.batch = args.batch

        self.converter_context = OnnxConverterContext(self.graph)

        # We can't run simplification and quantization overrides/custom ops as the simplification process
        # could possibly squash layers preventing the custom ops or quantization overrides from being used
        if not self.no_simplification and (args.quantization_overrides or args.custom_op_config_paths):
            self.no_simplification = True
            log_warning("Can't simplify the model when custom ops or quantization overrides are specified, converting without simplification.")

    def evaluate(self, model):
        """
        Performs a dry-run of the Onnx Model without actually converting it, highlighting potential issues with
        attributes, inputs/outputs or opset versions.
        :param model: An Onnx model
        :return:
        """
        from qti.aisw.converters.onnx import model_evaluator
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            log_warning("Potential errors found in {} as per Onnx's in-built checker tool".format(self.input_model_path))
            log_warning("{}: {}", type(e), str(e))
        log_info('Proceeding with model evaluation...................................\n')
        model_evaluator.setup_dry_run(model, self.dry_run)

    def convert(self):
        # Fork a process to run onnx simplifier to avoid segmentation fault, or some unexpected error
        def fork_process_run_simplifier(model, **kwargs):
            def run_simplifier(shared_queue, model, input_dims: dict = None, skip_optimization: bool = False):
                next_action_info = "trying simplification with skipped optimization" if skip_optimization is False else \
                                   "resuming normal conversion with unsimplified model"
                try:
                    if input_dims:
                        model_optimized, check_ok = onnxsim.simplify(model,
                                                                     input_shapes=input_dims,
                                                                     perform_optimization=not skip_optimization)
                    else:
                        model_optimized, check_ok = onnxsim.simplify(model,
                                                                     perform_optimization=not skip_optimization)
                    if check_ok:
                        log_info("Successfully simplified the onnx model in child process")
                        shared_queue.put(model_optimized)
                    else:
                        log_warning("Check failed. Couldn't simplify the onnx model, {}".format(next_action_info))
                except Exception as e:
                    log_warning("Model simplification failed with unexpected exception, {}".format(next_action_info))

            model_optimized = None
            with multiprocessing.Manager() as manager:
                shared_queue = manager.Queue()
                process = multiprocessing.Process(target=run_simplifier, args=(shared_queue, model), kwargs=kwargs)
                process.start()
                process.join()

                try:
                    model_optimized = shared_queue.get(block=False)
                    log_info("Successfully receive the simplified onnx model in main process")
                except queue.Empty as e:
                    next_action_info = "trying simplification with skipped optimization" if not kwargs.get("skip_optimization", False) else \
                                       "resuming normal conversion with unsimplified model"
                    # The child process was terminated by the signal segfault during simplification
                    if process.exitcode == -signal.SIGSEGV:
                        log_warning("Segmentation fault occur when running onnx-simplifier, " + next_action_info)
                    # The child process was terminated by a certain signal during simplification
                    elif process.exitcode != 0:
                        log_warning("Unexpected error occur when running onnx-simplifier, " + next_action_info)
            return model_optimized

        # Fork a process to run onnx infer_shape to avoid segmentation fault, or some unexpected error
        def fork_process_run_shape_inference(model, dump_inferred_model: bool = False, **kwargs):
            def run_shape_inference(shared_queue, model, dump_inferred_model: bool = False):
                try:
                    from onnx import shape_inference
                    inferred_model = shape_inference.infer_shapes(model)
                    shared_queue.put(inferred_model)
                except:
                    if dump_inferred_model:
                        log_error("Unable to dump inferred model since ONNX shape inference failed.")
                    else:
                        log_warning("ONNX shape inference failed.")

            inferred_model = None
            with multiprocessing.Manager() as manager:
                shared_queue = manager.Queue()
                process = multiprocessing.Process(target=run_shape_inference, args=(shared_queue, model), kwargs=kwargs)
                process.start()
                process.join()

                try:
                    inferred_model = shared_queue.get(block=False)
                except queue.Empty as e:
                    if dump_inferred_model:
                        log_error("Unable to dump inferred model since ONNX shape inference failed.")
                    else:
                        log_warning("ONNX shape inference failed.")

            return inferred_model


        model = onnx.load(self.input_model_path)

        # Before we do anything process the batch and input symbol overrides to ensure resolved input
        # symbols and updated batch values for shape inference and model simplification
        if self.batch or self.define_symbols:
            self._update_batch_and_symbols(model)

        # Try to simplify the model first
        if not self.no_simplification:
            try:
                import onnxsim
                dims_dict = {}
                if self.input_names and self.input_dims:
                    for i in range(len(self.input_names)):
                        dims_dict[self.input_names[i]] = [int(k) for k in self.input_dims[i].split(',')]

                model_optimized = fork_process_run_simplifier(model, input_dims=dims_dict)
                if not model_optimized:
                    model_optimized = fork_process_run_simplifier(model, input_dims=dims_dict, skip_optimization=True)

                if model_optimized:
                    model = model_optimized
            except ImportError as e:
                log_warning("Couldn't import onnx-simplifier. ({}: {})", type(e), str(e))
                log_warning("Install the onnx-simplifier for better model compatibility: \"pip3 install onnx-simplifier\"")
            except Exception as e:
                log_warning("Unknown error ({}: {}) during forking the process to run onnx-simplifier", type(e), str(e))

        # Enable faulthandler to dump the trace if segmentation fault occur
        faulthandler.enable()

        self.op_info.set_global_op_ver(model)

        if self.dry_run:
            self.evaluate(model)
            sys.exit(0)

        model_inferred = fork_process_run_shape_inference(model, self.dump_inferred_model)
        if model_inferred:
            model = model_inferred

        if self.input_dims and self.input_names:
            self._update_input_node(model)

        self.converter_context.weights = WeightProvider(model)
        self.converter_context.tensor_to_np_dtype  = self._track_tensor_type(model.graph)

        if self.output_names:
            # Trims the existing graph to the output nodes specified
            self._update_output_nodes(model)
        elif model.graph.output:
            # Add the Onnx model outputs to IR Graph
            for value_info in model.graph.output:
                self.graph.output_names.append(str(value_info.name))

        if self.graph.preserve_io_datatype_passed:
            # --custom_io has higher precedence than --preserve_io. Skip the tensors for which dtype is
            # supplied using the --custom_io option.
            tensors_having_custom_dtypes = []
            if self.graph.user_custom_io:
                for entry in self.graph.user_custom_io:
                    if "Datatype" in entry:
                        tensors_having_custom_dtypes.append(str(entry['IOName']))

            for arg in self.graph.preserve_io:
                if self.graph.preserve_io_datatype_passed == 1 and arg[0] == 'datatype':
                    for buffer_name in arg[1:]:
                        if buffer_name not in tensors_having_custom_dtypes:
                            self.graph.preserve_datatype_tensors[buffer_name] = None

            # self.graph.preserve_io_datatype_passed = 1 indicates that user intends to preserve datatype only for the specified tensors
            # self.graph.preserve_io_datatype_passed = 2 indicates that user intends to preserve datatype for all the input and output tensors
            for value_info in model.graph.input:
                if ((self.graph.preserve_io_datatype_passed == 1 and value_info.name in self.graph.preserve_datatype_tensors) or \
                    self.graph.preserve_io_datatype_passed == 2) and value_info.name not in tensors_having_custom_dtypes:
                    if value_info.type.tensor_type.elem_type == TensorProto.INT64:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(np.dtype('int64'))
                    else:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type))

            for value_info in model.graph.output:
                if ((self.graph.preserve_io_datatype_passed == 1 and value_info.name in self.graph.preserve_datatype_tensors) or \
                    self.graph.preserve_io_datatype_passed == 2) and value_info.name not in tensors_having_custom_dtypes:
                    if value_info.type.tensor_type.elem_type == TensorProto.INT64:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(np.dtype('int64'))
                    else:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type))

            # Throw an error if there is a conflict between the dtype passed using the --input_dtype option and the original dtype
            for k in self.graph.input_dtypes_dict:
                if k in self.graph.preserve_datatype_tensors and self.graph.input_dtypes_dict[k] != self.graph.preserve_datatype_tensors[k]:
                    log_error("Datatype mismatch for tensor %s. %s datatype set with --input_dtype and %s datatype set with --preserve_io!" \
                            % (k, str(self.graph.input_dtypes_dict[k]), self.graph.preserve_datatype_tensors[k]))
                    sys.exit(-1)

            for k in self.graph.preserve_datatype_tensors:
                if self.graph.preserve_datatype_tensors[k] == None:
                    log_error("Graph does not have the tensor %s" % (k))
                    sys.exit(-1)

        # Dumps the trimmed and inferred model, if it was requested
        if self.dump_inferred_model:
            inferred_model_filename = self.input_model_path.split('.')[0] + "_inferred.onnx"
            onnx.save(model, inferred_model_filename)

        # Dumps the value_info field of the ONNX graph after trimming, for debugging purposes
        if self.dump_value_info and model.graph.value_info:
            original_stdout = sys.stdout
            with open(self.input_model_path.split('.')[0] + "_value_info.info", "w") as file:
                sys.stdout = file
                print(model.graph.value_info)
                sys.stdout = original_stdout
        elif self.dump_value_info:
            log_warning("Unable to dump value info because field is not populated.")

        # populate custom op nodes if config paths are provided; condition is checked in function
        self.populate_custom_op_collection(model, 'onnx', converter_op_package_lib=self.converter_op_package_lib)

        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(converter_type("input", "onnx"),
                                                 onnx_translations.OnnxTranslationBase.ADD_INPUT_OP, value_info, self.graph)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = converter_type(src_op.op_type, "onnx")

            try:
                # first check if layer is a registered custom op in an op collection.
                # If so, the layer is added and the outer loop continues.
                if self.custom_op_factory and src_op.op_type in self.custom_op_factory.op_collection:
                    src_type = converter_type('custom', "onnx")
                    node = self.translations.apply_method_to_op(src_type,
                                                                onnx_translations.OnnxTranslationBase.ADD_OP,
                                                                src_op,
                                                                self.converter_context)
                    self.graph.add_src_op_info(node.op.name, [i for i in src_op.input], [o for o in src_op.output])

                elif src_op.domain in ['org.pytorch._caffe2']:
                    src_type = converter_type(src_op.op_type, "onnx_caffe2")
                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)

                else:
                    # If the op is not a custom operation, check the version and use the
                    # native converter translation
                    supported_version = self.translations.apply_method_to_op(src_type,
                                                                             onnx_translations.OnnxTranslationBase.SUPPORTED_VERSION,
                                                                             src_op.op_type)
                    self.op_info.validate_op_ver(src_op, supported_version)

                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        self.graph.eval_macs_params()

        if self.dump_custom_io_config_template:
            axis_format_to_layout_dict = {AxisTracker.AxisFormat.NDHWC: InputLayout.NDHWC,
                                        AxisTracker.AxisFormat.NCDHW: InputLayout.NCDHW,
                                        AxisTracker.AxisFormat.NSC: InputLayout.NHWC,
                                        AxisTracker.AxisFormat.NCS: InputLayout.NCHW,
                                        AxisTracker.AxisFormat.NFC: InputLayout.NFC,
                                        AxisTracker.AxisFormat.NCF: InputLayout.NCF,
                                        AxisTracker.AxisFormat.NTF: InputLayout.NTF,
                                        AxisTracker.AxisFormat.TNF: InputLayout.TNF,
                                        AxisTracker.AxisFormat.NF: InputLayout.NF,
                                        AxisTracker.AxisFormat.NC: InputLayout.NC,
                                        AxisTracker.AxisFormat.NONTRIVIAL: InputLayout.NONTRIVIAL}

            yaml_data = []
            comments = "# Custom I/O configuration template for the provided model.\n\n" \
                "# Layout field (optional) has two sub fields : Model and Custom. \n" \
                "# Model: Specify the layout of the buffer in the original model. Default value is obatained from the model \n" \
                "#        This is equivalent to the --input_layout option and both cannot be used together. \n" \
                "# Custom: Specify the custom layout desired for the buffer. Needs to be filled by the user. \n" \
                "# Model and Custom fields support valid QNN Layout. Accepted values are:\n" \
                "# NCDHW, NDHWC, NCHW, NHWC, NFC, NCF, NTF, TNF, NF, NC, F, NONTRIVIAL\n" \
                "# where, N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n\n" \
                "# Datatype field (optional) supports float32, float16 and uint8 datatypes. Default values for input buffer are obtained from the model \n" \
                "# This field is left empty for the output buffers. \n\n" \
                "# QuantParam field (optional) has three sub fields: Type, Scale and Offset \n" \
                "# Type: Set to QNN_DEFINITION_DEFINED (default) if the scale and offset are provided by the user else set to QNN_DEFINITION_UNDEFINED \n" \
                "# Scale and Offset fields are populated with dummy values as part of this template. Scale and Offset fields will be ignored for an I/O \n" \
                "# if the precision field corresponding to that I/O is not set to uint8 \n\n\n" \
                "# Model Inputs"

            yaml_data.append(comments)
            supported_datatypes = [np.float32, np.float16, np.uint8, np.int8, np.int32, np.uint32]

            for node in self.graph.get_input_nodes_to_graph():
                for buffer_name in node.output_names:
                    if self.converter_context.tensor_to_np_dtype[buffer_name] not in supported_datatypes:
                        continue
                    io_str = " - IOName: " + buffer_name + "\n"
                    io_str += "   Layout:\n     Model: " + axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n     Custom: " +  axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n"
                    io_str += "   Datatype: " + str(self.converter_context.tensor_to_np_dtype[buffer_name]) + "\n"
                    io_str += "   QuantParam:\n     Type: QNN_DEFINITION_DEFINED\n     Scale: 1.0\n     Offset: 0\n"
                    yaml_data.append(io_str)

            yaml_data.append("\n# Model Outputs")

            for node in self.graph.get_output_nodes_of_graph():
                for buffer_name in node.output_names:
                    if self.converter_context.tensor_to_np_dtype[buffer_name] not in supported_datatypes:
                        continue
                    io_str = " - IOName: " + buffer_name + "\n"
                    io_str += "   Layout:\n     Model: " + axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n     Custom: " +  axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n"
                    io_str += "   Datatype: " + str(self.converter_context.tensor_to_np_dtype[buffer_name]) + "\n"
                    io_str += "   QuantParam:\n     Type: QNN_DEFINITION_DEFINED\n     Scale: 1.0\n     Offset: 0\n"
                    yaml_data.append(io_str)

            f = open(self.dump_custom_io_config_template, 'w')
            f.write('\n'.join(yaml_data))
            f.close()
            sys.exit(0)

        return self.graph

    def _track_tensor_type(self, graph):
        tensor_to_np_dtype = {}

        for value_info in graph.input:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.value_info:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.output:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        return tensor_to_np_dtype

    def _update_batch_and_symbols(self, model):
        graph = model.graph
        if version.parse(onnx.version.version) < version.parse('1.6.0'):
            raise ValueError("--batch and --define_symbol commands not supported with ONNX versions < 1.6.0")

        # Override any symbols present in the input shapes with values passed by the client
        original_inputs = {node.name : node for node in graph.input}
        new_inputs = get_all_dims_info(graph.input)
        for name, dtype, dims in new_inputs:
            log_debug1('Proccessing overrides for input {} with dims {}'.format(name, dims))
            modified = False
            if self.define_symbols:
                for i, dim in enumerate(dims):
                    if isinstance(dim, str) and dim in self.define_symbols:
                        log_debug1('Overriding "{}" with {}'.format(dim, int(self.define_symbols[dim])))
                        dims[i]= int(self.define_symbols[dim])
                        modified = True

            # Override the batch dimension of all inputs with the passed value
            # TODO At some point make this batch logic common for all converters
            if self.batch:
                log_debug1('Overriding batch dim of {} from {} to {}'.format(name, dims[0], int(self.batch)))
                dims[0] = int(self.batch)
                modified = True

            # Remove the original input and add the new updated input
            if modified:
                new_input = onnx.helper.make_tensor_value_info(name, dtype, dims)
                log_debug1('Generated new input {} with dims {}'.format(name, dims))
                graph.input.remove(original_inputs[name])
                graph.input.append(new_input)

    def _update_input_node(self, model):
        graph = model.graph
        if version.parse(onnx.version.version) < version.parse('1.6.0'):
            raise ValueError("--input_dim command not supported with ONNX versions < 1.6.0")
        input_names = list(self.input_names)
        input_dims = list(self.input_dims)
        initializers = [node.name for node in graph.initializer]
        original_inputs = {node.name : node for node in graph.input}
        original_types = {node.name : node.type.tensor_type.elem_type for node in graph.input}
        new_inputs = {name: dim for name, dim in zip(input_names, input_dims)}

        # Step 1: remove original graph inputs
        for node_name in original_inputs:
            if node_name not in initializers:
                graph.input.remove(original_inputs[node_name])

        # Step 2: If input specified is part of graph inputs, update its dimensions
        for name in new_inputs:
            if name in initializers:
                raise ValueError("--input_dim command not supported with initializer " + name)
            elif name in original_inputs:
                dim = new_inputs[name]
                dims = tuple(map(int, dim.split(',')))
                type_new = original_types[name]
                input_new = onnx.helper.make_tensor_value_info(name,
                    type_new, dims)
                graph.input.append(input_new)
                input_names.remove(name)
                input_dims.remove(dim)
            else:
                continue

        # Check if all inputs are accounted for, if Yes nothing more to be done. Return
        if len(input_names) == 0 and len(input_dims) == 0:
            return

        # Get the type of each model input
        input_types = {}
        for input_name in input_names:
            input_found, input_type, _ = get_type_dims_info(model.graph.input, input_name)
            input_types[input_name] = input_type if input_found else onnx.TensorProto.FLOAT

        # Step 3: If input specified is intermittent graph output,
        #         a.  Add this buffer to a list for removal later
        #         b.  Create input TensorProto with this name and dimension
        bufs_to_remove = set()
        for i, src_op in enumerate(graph.node):
            for output_buf_name in src_op.output:
                if output_buf_name in input_names:
                    position = input_names.index(output_buf_name)
                    dim = input_dims[position]
                    dims = tuple(map(int, dim.split(',')))
                    input_new = onnx.helper.make_tensor_value_info(output_buf_name, input_types[output_buf_name], dims)
                    graph.input.append(input_new)
                    bufs_to_remove.add(output_buf_name)
                    input_names.remove(output_buf_name)
                    input_dims.remove(dim)

        # Check if all inputs specified are accounted for
        if len(input_names) != 0 and len(input_dims) != 0:
            invalid_names = ", ".join(input_names)
            raise ValueError("--input_dim command input name(s) not found: {}".format(invalid_names))

        # Step 4: Find all nodes to be removed from the graph. These include:
        #   a. Nodes that produce the buffers cached for removal
        #   b. All nodes that precede them in the graph
        nodes_to_remove = []
        while bufs_to_remove:
            buf_name = bufs_to_remove.pop()
            if buf_name in original_inputs or buf_name in initializers:
                # This was already removed or does not need to be handled
                continue

            # Find node that produces the buffer or is named after the buffer
            node_list = [node for node in graph.node if buf_name in node.output]
            if not node_list:
                raise KeyError("Node that produces {} not found".format(buf_name))
            elif len(node_list) != 1:
                raise KeyError("Multiple nodes {} found for output buffer {}".format(node_list, buf_name))

            node = node_list[0]
            # Add all inputs of this node as also to be removed
            bufs_to_remove.update(set(node.input))
            # Add this node to be removed if not already added
            if node not in nodes_to_remove:
                nodes_to_remove.append(node)

        # Step 5: Remove the nodes marked in Step 4
        # Check that all buffers in a slice were specified, if not Throw Error
        remaining_nodes = [node for node in graph.node if node not in nodes_to_remove]
        remaining_buffers = set()
        for remaining_node in remaining_nodes:
            remaining_buffers.update(remaining_node.input)
        for node in nodes_to_remove:
            for output in node.output:
                if output in remaining_buffers and output not in self.input_names:
                    raise ValueError("Cannot disconnect node with outputs: {} as output buffer"
                                     ": {} is still in use and was not specified as input to the Model".format
                                     (str(node.output), str(output)))
            graph.node.remove(node)

    def _update_output_nodes(self, model):

        # Determine which nodes should be retained
        nodes_to_retain = []
        queue = list(self.output_names)
        visited = set(queue)
        while queue:
            input_name = queue.pop(0)
            preceding_nodes = [node for node in model.graph.node if input_name in node.output]
            for node in preceding_nodes:
                nodes_to_retain.append(node)
                for input_name in node.input:
                    if input_name in visited:
                        continue
                    queue.append(input_name)
                    visited.add(input_name)

        # Remove nodes that are not retained
        for node in [node for node in model.graph.node if node not in nodes_to_retain]:
            model.graph.node.remove(node)

        # Get the output dimensions of the new output nodes
        new_output_value_infos = []
        for output_name in self.output_names:
            # First check the graph outputs for info on outputs
            output_found, output_type, output_dims = get_type_dims_info(model.graph.output, output_name)

            # Fallback to using optional value_info field for info on new outputs
            if not output_found and model.graph.value_info:
                output_found, output_type, output_dims = get_type_dims_info(model.graph.value_info, output_name)

            # Finally, fallback to using graph inputs for info on new outputs
            if not output_found:
                output_found, output_type, output_dims = get_type_dims_info(model.graph.input, output_name)

            if output_found:
                output_value_info = onnx.helper.make_tensor_value_info(output_name, output_type, output_dims)
            else:
                output_value_info = onnx.helper.ValueInfoProto()
                output_value_info.name = output_name

            new_output_value_infos.append(output_value_info)

        # Remove old output nodes
        for output_node in [_ for _ in model.graph.output]:
            model.graph.output.remove(output_node)

        # Add new output info
        model.graph.output.extend(new_output_value_infos)


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if hasattr(op, 'LEGACY_TRANSLATION_KEY'):
            name_prefix_str = str(op.LEGACY_TRANSLATION_KEY)
        else:
            name_prefix_str = str(op.type)
        if op.name:
            return str(op.name)
        elif op.type == 'custom':
            return "%s_%s_%d" % (str(op.custom_type), name_prefix_str, count)
        else:
            return "%s_%d" % (name_prefix_str, count)

    def get_op_name_by_type(self, op_type, legacy_translation_key, custom_op_type=""):
        count = self.type_count.get(op_type, 0)
        self.type_count[op_type] = count + 1
        if legacy_translation_key:
            name_prefix_str = str(legacy_translation_key)
        else:
            name_prefix_str = str(op_type)
        if custom_op_type:
            return "%s_%s_%d" % (str(custom_op_type), name_prefix_str, count)
        else:
            return "%s_%d" % (name_prefix_str, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     onnx_translations.OnnxTranslationBase.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
