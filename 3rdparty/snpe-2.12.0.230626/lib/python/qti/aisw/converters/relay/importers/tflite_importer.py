# ==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.relay.passes.pattern_match import tflite_detection_postprocess, tflite_dequantize
from qti.aisw.converters.relay.utils import get_key_from_expr
from .relay_importer import RelayImporter, RelaySpanParser
import tvm
from tvm.relay.frontend import tflite as tflite_to_relay
from tvm.relay.build_module import bind_params_by_name

# TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
try:
    import tflite
except TypeError:
    import tflite.Model as tflite


class TFLiteImporter(RelayImporter):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(TFLiteImporter.ArgParser, self).__init__(conflict_handler='resolve', **kwargs)
            self.add_required_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The names and dimensions of the network input layers specified "
                                            "in the format [input_name comma-separated-dimensions], "
                                            "for example: \n"
                                            "    'data' 1,224,224,3\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dim on the command "
                                            "line like: \n"
                                            "    --input_dim 'data1' 1,224,224,3 --input_dim 'data2' 1,50,100,3")
            self.add_optional_argument('--input_dtype', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DTYPE'),
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

    def __init__(self, args):
        super(TFLiteImporter, self).__init__(args)

        self.shape_dict = {}
        for in_name, in_dims in args.input_dim:
            self.shape_dict[in_name] = [int(i) for i in in_dims.split(',')]

        if args.input_dtype:
            self.dtype_dict = {in_name: in_dtype for in_name, in_dtype in args.input_dtype}
        else:
            self.dtype_dict = {}
            for input_name in self.shape_dict:
                if input_name not in self.dtype_dict:
                    self.dtype_dict[input_name] = "float32"

        # register custom relay ops
        self._register_ops()

    def convert_to_relay(self, input_model_path, **kwargs):
        if isinstance(input_model_path, str):
            tflite_model_buf = open(input_model_path, "rb").read()
        elif isinstance(input_model_path, bytes):
            tflite_model_buf = input_model_path
        else:
            raise TypeError("Unsupported type {} for {}".format(type(input_model_path), input_model_path))
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        self.mod, self.params = tflite_to_relay.from_tflite(tflite_model, self.shape_dict, self.dtype_dict)

        self._post_process()
        return self.mod, self.params, self.expr_to_source_info_dict

    def _post_process(self):
        """post-process Relay module, including necessary fixes and optimizations"""

        def _populated_encodings_dict(span_to_encodings):
            def visit_module(expr):
                if hasattr(expr, 'span'):
                    key = get_key_from_expr(expr)
                    if expr.span in span_to_encodings:
                        self.expr_to_source_info_dict[key].set_encodings(span_to_encodings[expr.span])
            tvm.relay.analysis.post_order_visit(self.mod["main"], visit_module)

        # bind TVM params variance to const
        self.mod["main"] = bind_params_by_name(self.mod["main"], self.params)

        # use span as key to record encodings and populate them after other transform pass since span will not change
        # e.g.,
        # %1 = Call(nn.conv2d(...), span=spanA, checked_type=undefined), => hash_value1, span=spanA
        # => after InferType transform pass
        # %1 = Call(nn.conv2d(...), span=spanA, checked_type=defined), => hash_value2 (changed), span=spanA (not change)
        span_to_encodings = {}

        # Prepare for Relay Passes
        seq = tvm.transform.Sequential([
            tflite_dequantize.DequantizePass(self.dtype_dict, span_to_encodings),
            # compress detection_postprocess expression back to one ir
            tflite_detection_postprocess.IdentifyTFLiteDetectionPostProcess(),
            tvm.relay.transform.FoldConstant(),
            tvm.relay.transform.InferType(),
        ])

        # need opt_level=3 to trigger ConvertLayout
        with tvm.transform.PassContext(opt_level=3):
            self.mod = seq(self.mod)
        RelaySpanParser(self.expr_to_source_info_dict).visit(self.mod['main'])
        _populated_encodings_dict(span_to_encodings)

    # TODO: revisit if we should put this functionality to another module or package.
    # Current put it here just because there are not so many OPs we need to register.
    @staticmethod
    def _register_ops():
        tflite_detection_postprocess.register_op()
