# ==============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import traceback
from abc import abstractmethod, ABC
import qti.aisw.converters.common.converter_ir.op_graph as op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder
from qti.aisw.converters.common.converter_ir.op_policies import ConversionNamePolicy
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.validation_utils import *
from qti.aisw.converters.common.common_base import ConverterBase


class ConverterFrontend(ConverterBase, ABC):

    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, **kwargs):
            super(ConverterFrontend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--out_node', '--out_name', type=str, dest='out_names', action='append', default=[],
                                       help="Name of the graph\'s output Tensor Names. Multiple output names "
                                            "should be provided separately like: \n"
                                            "    --out_name out_1 --out_name out_2")
            self.add_optional_argument('--input_type', "-t", nargs=2, action='append',
                                       help='Type of data expected by each input op/layer. Type for each input '
                                            'is |default| if not specified. For example: "data" image.Note that '
                                            'the quotes should always be included in order to handle special '
                                            'characters, spaces,etc. For multiple inputs specify multiple '
                                            '--input_type on the command line.\n'
                                            'Eg: \n'
                                            '   --input_type "data1" image --input_type "data2" opaque \n'
                                            'These options get used by DSP runtime and following descriptions '
                                            'state how input will be handled for each option.\n'
                                            'Image: \n'
                                            'Input is float between 0-255 and the input\'s mean is 0.0f '
                                            'and the input\'s max is 255.0f. We will cast the float to uint8ts '
                                            'and pass the uint8ts to the DSP. \n'
                                            'Default: \n'
                                            'Pass the input as floats to the dsp directly and the DSP '
                                            'will quantize it.\n'
                                            'Opaque: \n'
                                            'Assumes input is float because the consumer layer(i.e next '
                                            'layer) requires it as float, therefore it won\'t be quantized.\n'
                                            'Choices supported:\n   ' + '\n   '.join(op_graph.InputType.
                                                                                     get_supported_types()),
                                       metavar=('INPUT_NAME', 'INPUT_TYPE'), default=[])

            self.add_optional_argument('--input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'), default=[],
                                       help="The names and datatype of the network input layers specified "
                                            "in the format [input_name datatype], "
                                            "for example: \n"
                                            "    'data' 'float32'\n"
                                            "Default is float32 if not specified\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dtype on the command "
                                            "line like: \n"
                                            "    --input_dtype 'data1' 'float32' --input_dtype 'data2' 'float32'")

            self.add_optional_argument('--input_encoding', "-e", nargs='+', action='append',
                                       help='Usage: '
                                            '    --input_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT]\n'
                                            'Input encoding of the network inputs. Default is bgr. \n'
                                            'e.g.\n'
                                            '   --input_encoding "data" rgba \n'
                                            'Quotes must wrap the input node name to handle special characters, \n'
                                            'spaces, etc. To specify encodings for multiple inputs, invoke \n'
                                            '--input_encoding for each one. \n'
                                            'e.g.\n'
                                            '    --input_encoding "data1" rgba --input_encoding "data2" other\n'
                                            'Optionally, an output encoding may be specified for an input node by \n'
                                            'providing a second encoding. The default output encoding is bgr.\n'
                                            'e.g. \n'
                                            '    --input_encoding "data3" rgba rgb \n'
                                            'Input encoding types:\n '
                                            '    image color encodings: bgr,rgb, nv21, nv12, ... \n'
                                            '    time_series: for inputs of rnn models; \n'
                                            '    other: not available above or is unknown. \n'
                                            'Supported encodings:\n   ' + '\n   '.join(op_graph.InputEncodings.
                                                                                       get_supported_encodings()),
                                       metavar="\b", default=[])

            self.add_optional_argument('--input_layout', "-l", nargs=2, action='append',
                                       help='Layout of each input tensor. If not specified, it will use the default\n'
                                            'based on the Source Framework, shape of input and input encoding.\n'
                                            'Accepted values are-\n'
                                            '    ' + ', '.join(op_graph.InputLayout.get_supported_types()) + '\n'
                                            'N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n'
                                            'NDHWC/NCDHW used for 5d inputs\n'
                                            'NHWC/NCHW used for 4d image-like inputs\n'
                                            'NFC/NCF used for inputs to Conv1D or other 1D ops\n'
                                            'NTF/TNF used for inputs with time steps like the ones used for LSTM op\n'
                                            'NF used for 2D inputs, like the inputs to Dense/FullyConnected layers\n'
                                            'NC used for 2D inputs with 1 for batch and other for Channels (rarely used) \n'
                                            'F used for 1D inputs, e.g. Bias tensor\n'
                                            'NONTRIVIAL for everything else'
                                            'For multiple inputs specify multiple --input_layout on the command line.\n'
                                            'Eg: \n'
                                            '    --input_layout "data1" NCHW --input_layout "data2" NCHW \n',
                                       metavar=('INPUT_NAME', 'INPUT_LAYOUT'), default=[])

            self.add_optional_argument('--custom_io', type=str, default="",
                                       help='Use this option to specify a yaml file for custom IO.')
            self.add_optional_argument('--preserve_io', nargs='*', action='append', default=[],
                                       help='Use this option to preserve IO layout and datatype. The different ways of using this option are as follows:\n'
                                            '    --preserve_io layout <space separated list of names of inputs and outputs of the graph>\n'
                                            '    --preserve_io datatype <space separated list of names of inputs and outputs of the graph>\n'
                                            'In this case, user should also specify the string - layout or datatype in the command to indicate that converter needs to\n'
                                            'preserve the layout or datatype. e.g.\n'
                                            '   --preserve_io layout input1 input2 output1 \n'
                                            '   --preserve_io datatype input1 input2 output1 \n'
                                            'Optionally, the user may choose to preserve the layout and/or datatype for all the inputs and outputs of the graph.\n'
                                            'This can be done in the following two ways:\n'
                                            '    --preserve_io layout\n'
                                            '    --preserve_io datatype\n'
                                            'Additionally, the user may choose to preserve both layout and datatypes for all IO tensors by just passing the option as follows:\n'
                                            '    --preserve_io\n'
                                            'Note: Only one of the above usages are allowed at a time.\n'
                                            'Note: --custom_io gets higher precedence than --preserve_io.\n')

            q_group = self.add_argument_group(title='Quantizer Options')
            q_group.add_argument('--quantization_overrides', type=str, default="",
                                 help='Use this option to specify a json file with parameters to use for '
                                      'quantization. These will override any quantization data carried from conversion '
                                      '(eg TF fake quantization) or calculated during the normal quantization process. '
                                      'Format defined as per AIMET specification.')
            q_group.add_argument('--keep_quant_nodes', default=False, action="store_true",
                                 help='Use this option to keep activation quantization nodes in the graph rather than '
                                      'stripping them.')

    def __init__(self, args,
                 naming_policy=ConversionNamePolicy(),
                 shape_inference_policy=None,
                 axis_order=AxisOrder(),
                 custom_op_factory=None):
        super(ConverterFrontend, self).__init__(args)
        self.output_names = args.out_names

        for input_encoding in args.input_encoding:
            if len(input_encoding) not in [2, 3]:
                raise ValueError('Received incorrect number of input encodings for input {}. Got {}, expected \n'
                                 'one input encoding and one (optional) output encoding per graph input in the \n'
                                 'following format: \n'
                                 '    --input_encoding "INPUT_NAME" INPUT_ENCODING_IN [INPUT_ENCODING_OUT] \n'
                                 .format(input_encoding[0], len(input_encoding) - 1))

        self.graph = op_graph.IROpGraph(naming_policy, shape_inference_policy,
                                        args.input_type, args.input_dtype, args.input_encoding,
                                        axis_order,
                                        input_layouts=args.input_layout,
                                        quantization_overrides=args.quantization_overrides,
                                        custom_io=args.custom_io,
                                        preserve_io=args.preserve_io,
                                        keep_quant_nodes=args.keep_quant_nodes,
                                        output_nodes=self.output_names,
                                        keep_int64_inputs=self.keep_int64_inputs)

        self.custom_op_config_paths = args.custom_op_config_paths
        self.custom_op_factory = custom_op_factory
        self.converter_op_package_lib = args.converter_op_package_lib

    @abstractmethod
    def convert(self):
        """
        Convert the input framework model to IROpGraph: to be overridden by each framework
        """
        pass

    # TODO: Move once converter base hierarchy is refactored
    def populate_custom_op_collection(self,
                                      model,
                                      converter_type='onnx',
                                      **kwargs):
        if "converter_op_package_lib" in kwargs:
            kwargs["converter_op_package_libs"] = kwargs["converter_op_package_lib"].split(',')[::-1]
            for lib_path in kwargs["converter_op_package_libs"]:
                check_filename_encoding(lib_path)
                io_utils.check_validity(lib_path, is_path=True, must_exist=True)
        # Create a custom op collection based on configs provided by user
        if self.custom_op_config_paths is not None:
            for config_path in self.custom_op_config_paths:
                try:
                    self.custom_op_factory.parse_config(config_path,
                                                        model=model,
                                                        converter_type=converter_type,
                                                        **kwargs)
                except Exception as e:
                    if not is_log_level_debug():
                        traceback.print_exc()
                    log_error("Error populating custom ops from: {}\n {}".format(config_path,
                                                                                 str(e)))
                    sys.exit(-1)

                if not len(self.custom_op_factory.op_collection) and \
                        not self.custom_op_factory.default_op_collection:
                    raise LookupError("CUSTOM_OP_NOT_FOUND: "
                                      "None of the custom Ops present in the "
                                      "config were found in the provided model.")
