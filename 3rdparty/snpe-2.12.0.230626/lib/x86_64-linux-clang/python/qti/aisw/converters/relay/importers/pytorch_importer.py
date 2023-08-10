# ==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils.converter_utils import (
    log_debug1,
    log_warning,
)
from qti.aisw.converters.relay.passes.pattern_match.channel_shuffle import IdentifyChannelShuffle
from qti.aisw.converters.relay.passes.pattern_match.upsampling import IdentifyUpsampling
from .relay_importer import RelayImporter, RelaySpanParser
import torch
import torchvision
import tvm
import tvm.relay.op.op as _op
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.frontend import from_pytorch

class PyTorchImporter(RelayImporter):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(PyTorchImporter.ArgParser, self).__init__(conflict_handler='resolve', **kwargs)
            self.add_required_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The names and dimensions of the network input layers specified "
                                            "in the format [input_name comma-separated-dimensions], "
                                            "for example: \n"
                                            "    'data' 1,3,224,224\n"
                                            "Note that the quotes should always be included in order to handle"
                                            "special characters, spaces, etc. \n"
                                            "For multiple inputs specify multiple --input_dim on the command "
                                            "line like: \n"
                                            "    --input_dim 'data1' 1,3,224,224 --input_dim 'data2' 1,50,100,3")
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
        super(PyTorchImporter, self).__init__(args)

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

    def _post_process(self):
        """post-process Relay module, including necessary fixes and optimizations"""

        # register custom relay ops
        self._register_ops()

        # bind TVM params variance to const
        self.mod["main"] = bind_params_by_name(self.mod["main"], self.params)

        # Prepare for Relay Passes
        # Current only these these are reasonable to assign desired layouts.
        # Other OPs able to assign desired layouts are:
        #   nn.conv3d
        #   nn.deformable_conv2d
        #   vision.roi_align
        #   vision.roi_pool
        #   qnn.conv2d
        # Searched by Op having a callback decorated by "register_convert_op_layout"
        desired_layouts = {
            "nn.conv2d": ["NHWC", "HWIO"],
            "nn.conv2d_transpose": ["NHWC", "HWIO"],
            ### Set BN/IN's layout as "NSC" to extend the support to ND
            "nn.batch_norm": ["NSC"],
            "nn.instance_norm": ["NSC"],
            "nn.avg_pool2d": ["NHWC"],
            "nn.avg_pool3d": ["NDHWC"],
            "nn.max_pool2d": ["NHWC"],
            "nn.max_pool3d": ["NDHWC"],
            "nn.global_avg_pool2d": ["NHWC"],
            "nn.global_max_pool2d": ["NHWC"],
            "nn.adaptive_avg_pool2d": ["NHWC"],
            "nn.adaptive_max_pool2d": ["NHWC"],
            "image.resize": ["NHWC"],
            "channel_shuffle": ["NHWC"],
            "nn.upsampling": ["NHWC"],
            "nn.prelu": ["NHWC", "C"],
            "qnn.quantize": ["NHWC", "NHWC", "NHWC"],
            "qnn.dequantize": ["NHWC", "NHWC", "NHWC"],
        }
        seq = tvm.transform.Sequential([
            # PyTorch frontend assumes NCHW layout
            IdentifyChannelShuffle(data_layout="NCHW"),
            IdentifyUpsampling(data_layout="NCHW"),
            tvm.relay.transform.ConvertLayout(desired_layouts),
            tvm.relay.transform.FoldConstant(),
            tvm.relay.transform.SimplifyExpr(),
            tvm.relay.transform.InferType(),
        ])

        # need opt_level=3 to trigger ConvertLayout
        with tvm.transform.PassContext(opt_level=3):
            self.mod = seq(self.mod)
        RelaySpanParser(self.expr_to_source_info_dict).visit(self.mod['main'])

    # TODO: revisit if we should put this functionality to another module or package.
    # Current put it here just because there are not so many OPs we need to register.
    @staticmethod
    def _register_ops():

        # TVM label 4D tensors with letters 'N', 'C', 'H', and 'W',
        # 3D tensors with 'N'(batch), 'C'(channel), 'W'(width),
        # 5D tensors with 'N', 'C', 'D', 'H', 'W'.
        # To make layout_transform OP work, we follow TVM conventions.
        channel_first_data_layout_3d = tvm.tir.data_layout.layout("NCW")
        channel_last_data_layout_3d = tvm.tir.data_layout.layout("NWC")

        channel_first_data_layout_4d = tvm.tir.data_layout.layout("NCHW")
        channel_last_data_layout_4d = tvm.tir.data_layout.layout("NHWC")

        channel_first_data_layout_5d = tvm.tir.data_layout.layout("NCDHW")
        channel_last_data_layout_5d = tvm.tir.data_layout.layout("NDHWC")

        const_layout = tvm.tir.data_layout.layout("C")

        rank_to_channel_first_layout = {
            3: channel_first_data_layout_3d,
            4: channel_first_data_layout_4d,
            5: channel_first_data_layout_5d,
        }

        rank_to_channel_last_layout = {
            3: channel_last_data_layout_3d,
            4: channel_last_data_layout_4d,
            5: channel_last_data_layout_5d,
        }

        ##########################################################
        # Op BatchNorm
        ##########################################################

        @_op.register_convert_op_layout("nn.batch_norm")
        def convert_batch_norm(attrs, inputs, tinfos, desired_layouts):
            attrs = dict(attrs)
            input_shape = tinfos[0].shape

            attrs['axis'] = 1 if desired_layouts[0] == "NCS" else len(input_shape)-1
            # relay.nn.batch_norm is a tuple, will return first expression
            return relay.nn.batch_norm(*inputs, **attrs)[0]

        @tvm.ir.register_op_attr("nn.batch_norm", "FInferCorrectLayout", level=11)
        def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):
            input_rank = len(old_in_types[0].shape)
            if attrs.axis % input_rank == 1:
                desired_layout = rank_to_channel_first_layout[input_rank]
            else:
                desired_layout = rank_to_channel_last_layout[input_rank]

            return [[desired_layout, const_layout, const_layout, const_layout, const_layout],
                    [desired_layout, const_layout, const_layout]]

        ##########################################################
        # Op InstanceNorm
        ##########################################################

        @_op.register_convert_op_layout("nn.instance_norm")
        def convert_instance_norm(attrs, inputs, tinfos, desired_layouts):
            attrs = dict(attrs)
            input_shape = tinfos[0].shape

            attrs['axis'] = 1 if desired_layouts[0] == "NCS" else len(input_shape)-1
            return relay.nn.instance_norm(*inputs, **attrs)

        @tvm.ir.register_op_attr("nn.instance_norm", "FInferCorrectLayout", level=11)
        def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):
            input_rank = len(old_in_types[0].shape)
            if attrs.axis % input_rank == 1:
                desired_layout = rank_to_channel_first_layout[input_rank]
            else:
                desired_layout = rank_to_channel_last_layout[input_rank]

            return [[desired_layout, const_layout, const_layout],
                    [desired_layout,]]

        ##########################################################
        # Op channel_shuffle
        ##########################################################

        channel_shuffle_op_name = "channel_shuffle"
        _op.register(channel_shuffle_op_name)
        custom_op = _op.get(channel_shuffle_op_name)
        custom_op.set_num_inputs(1)

        custom_op.add_type_rel("Identity")
        _op.register_pattern(channel_shuffle_op_name, _op.OpPattern.INJECTIVE)

        @_op.register_convert_op_layout(channel_shuffle_op_name)
        def convert_layout(attrs, inputs, tinfos, desired_layouts):
          assert len(desired_layouts) == 1, "Only one desired layout is expected."
          assert len(inputs) == 1, "Number of inputs mismatched."
          new_attrs = dict(attrs)
          new_attrs["data_layout"] = desired_layouts[0]
          call_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
          return relay.Call(custom_op, inputs, call_attrs)

        @tvm.ir.register_op_attr(channel_shuffle_op_name, "FInferCorrectLayout")
        def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):
          layout = tvm.tir.data_layout.layout(attrs["data_layout"])
          return [[layout], [layout]]

        ##########################################################
        # Op upsampling
        ##########################################################

        upsampling_op_name = "nn.upsampling"

        @_op.register_convert_op_layout(upsampling_op_name)
        def convert_layout(attrs, inputs, tinfos, desired_layouts):
            assert len(desired_layouts) == 1, "Only one desired layout is expected."
            assert len(inputs) == 1, "Number of inputs mismatched."
            new_attrs = dict(attrs)
            new_attrs["layout"] = desired_layouts[0]
            return tvm.relay.op.nn.upsampling(inputs[0], **new_attrs)

        ##########################################################
        # Op prelu
        ##########################################################

        prelu_op_name = "nn.prelu"

        @_op.register_convert_op_layout(prelu_op_name)
        def convert_layout(attrs, inputs, tinfos, desired_layouts):
            assert len(inputs) == 2, f"expect 2 inputs for Prelu but got {len(inputs)}."

            org_axis = attrs.axis
            desired_data_layout = desired_layouts[0]
            data_tensor_rank = len(tinfos[0].shape)

            if "C" not in desired_data_layout:
                log_warning("C(channel) not in desired data layout {}", desired_data_layout)
                return tvm.relay.op.nn.prelu(inputs[0], inputs[1], org_axis)

            axis = org_axis
            if desired_data_layout[-1] == "C":
                # Channel dimension is the last dimension
                axis = data_tensor_rank - 1
            return tvm.relay.op.nn.prelu(inputs[0], inputs[1], axis)

        # level=11 to overwrite the original FInferCorrect which is level=10.
        @tvm.ir.register_op_attr(prelu_op_name, "FInferCorrectLayout", level=11)
        def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):

            data_layout = old_in_layouts[0]
            alpha_layout = tvm.tir.data_layout.layout("C")
            data_tensor_type = old_in_types[0]
            data_tensor_rank = len(data_tensor_type.shape)

            channel_axis = attrs.axis

            if channel_axis == data_tensor_rank-1:
                desired_layout = rank_to_channel_last_layout[data_tensor_rank]
            else:
                desired_layout = rank_to_channel_first_layout[data_tensor_rank]

            return [[desired_layout, alpha_layout], [desired_layout]]

        ##########################################################
        # Op qnn.quantize
        ##########################################################

        quantize_op_name = 'qnn.quantize'
        @_op.register_convert_op_layout(quantize_op_name)
        def convert_layout(attrs, inputs, tinfos, desired_layouts):
            assert len(inputs) == 3, f"expect 3 inputs for Quantize but got {len(inputs)}."
            attrs = dict(attrs)
            attrs['axis'] = desired_layouts[0].index('C')
            return relay.qnn.op.quantize(inputs[0], inputs[1], inputs[2], **attrs)

        # level=11 to overwrite the original FInferCorrect which is level=10.
        @tvm.ir.register_op_attr(quantize_op_name, "FInferCorrectLayout", level=11)
        def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):

            data_layout = old_in_layouts[0]
            data_tensor_type = old_in_types[0]
            data_tensor_rank = len(data_tensor_type.shape)

            # Pytorch always set the channel axis to the 2nd dimension.
            # We require the channel axis the last dimension.
            #
            # For 2D tensor, the last dimension is the channel dimension
            # so we don't do anything.
            # Also, we ignore tensor rank > 5 because TVM has no corresponding labels.
            if data_tensor_rank not in [3, 4, 5]:
                # Same as the original FInferCorrectLayout
                return [[data_layout, data_layout, data_layout], [data_layout]]

            channel_axis = attrs.axis
            if channel_axis < 0:
                channel_axis += data_tensor_rank

            if channel_axis == data_tensor_rank-1:
                desired_layout = rank_to_channel_last_layout[data_tensor_rank]
            else:
                desired_layout = rank_to_channel_first_layout[data_tensor_rank]

            if new_in_layouts is not None:
                return [[desired_layout, desired_layout, desired_layout], [new_in_layouts[0]]]
            else:
                return [[desired_layout, desired_layout, desired_layout], [old_in_layouts[0]]]

        ##########################################################
        # Op qnn.dequantize
        ##########################################################

        dequantize_op_name = 'qnn.dequantize'
        @_op.register_convert_op_layout(dequantize_op_name)
        def convert_layout(attrs, inputs, tinfos, desired_layouts):
            assert len(inputs) == 3, f"expect 3 inputs for Dequantize but got {len(inputs)}."
            attrs = dict(attrs)
            attrs['axis'] = desired_layouts[0].index('C')
            return relay.qnn.op.dequantize(inputs[0], inputs[1], inputs[2], **attrs)

        # level=11 to overwrite the original FInferCorrect which is level=10.
        @tvm.ir.register_op_attr(dequantize_op_name, "FInferCorrectLayout", level=11)
        def infer_correct_layout(attrs, new_in_layouts, old_in_layouts, old_in_types):

            data_layout = old_in_layouts[0]
            data_tensor_type = old_in_types[0]
            data_tensor_rank = len(data_tensor_type.shape)

            # Pytorch always set the channel axis to the 2nd dimension.
            # We require the channel axis the last dimension.
            #
            # For 2D tensor, the last dimension is the channel dimension
            # so we don't do anything.
            # Also, we ignore tensor rank > 5 because TVM has no corresponding labels.
            if data_tensor_rank not in [3, 4, 5]:
                # Same as the original FInferCorrectLayout
                return [[data_layout, data_layout, data_layout], [data_layout]]

            channel_axis = attrs.axis
            if channel_axis < 0:
                channel_axis += data_tensor_rank

            if channel_axis == data_tensor_rank-1:
                desired_layout = rank_to_channel_last_layout[data_tensor_rank]
            else:
                desired_layout = rank_to_channel_first_layout[data_tensor_rank]

            if new_in_layouts is not None:
                return [[desired_layout, desired_layout, desired_layout], [new_in_layouts[0]]]
            else:
                return [[desired_layout, desired_layout, desired_layout], [old_in_layouts[0]]]

        # TODO: More op registeries
        # If it's hard to maintain (eg. too much registries, need of values from outside, ... etc),        # consider re-organize to new files/directoy.

    def convert_to_relay(self, input_model_path, **kwargs):

        pytorch_model = torch.jit.load(input_model_path)

        shape_list = list()

        for k, v in self.shape_dict.items():
            shape_list.append((k, v))

        self.mod, self.params = from_pytorch(pytorch_model, input_model_path, shape_list, self.dtype_dict, use_parser_friendly_name=True)

        self._post_process()

        return self.mod, self.params, self.expr_to_source_info_dict
