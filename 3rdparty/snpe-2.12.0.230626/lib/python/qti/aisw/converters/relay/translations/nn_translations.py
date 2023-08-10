# ==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.utils.converter_utils import (
    converter_type,
    log_assert,
    log_debug1,
    log_debug3,
)
from qti.aisw.converters.common.converter_ir.op_adapter import (
    BatchnormOp,
    ChannelShuffleOp,
    ConstantOp,
    Conv2dOp,
    DepthToSpaceOp,
    DepthwiseConv2dOp,
    DetectionOutputOp,
    ElementwiseBinaryOp,
    FullyConnectedOp,
    IdentityOp,
    LayerNormOp,
    LogSoftmaxOp,
    MatMulOp,
    NeuronOp,
    PadOp,
    Pool2dOp,
    PreluOp,
    ResizeOp,
    SoftmaxOp,
    SpaceToDepthOp,
    TransposeConv2dOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase, validate_const_name
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay


# ------------------------------------------------------------------------------
#   Adaptive Average Pool2D
# ------------------------------------------------------------------------------
class RelayAdaptiveAvgPool2DTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayAdaptiveAvgPool2DTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        adaptive_avg_pool_attr = relay_expr.attrs
        attr_dict['layout'] = adaptive_avg_pool_attr.layout
        attr_dict["output_size"] = adaptive_avg_pool_attr.output_size if hasattr(adaptive_avg_pool_attr, 'output_size') else None

        log_debug3("\tlayout {}", attr_dict['layout'])
        log_debug3("\toutput_size {}", attr_dict['output_size'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, Pool2dOp.TRANSLATION_KEY, Pool2dOp.LEGACY_TRANSLATION_KEY)
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        data_layout = attr_dict['layout']
        output_size = attr_dict['output_size']

        if data_layout != "NHWC":
            raise ValueError("No support {} data layout".format(data_layout))

        if output_size is None:
            size_y = input_shape[1]
            size_x = input_shape[2]
            stride_y = 1
            stride_x = 1
        else:
            h = input_shape[1]
            w = input_shape[2]
            output_size_h = int(output_size[0])
            output_size_w = int(output_size[1]) if len(output_size) == 2 else int(output_size[0])
            stride_y = int(h / output_size_h)
            stride_x = int(w / output_size_w)
            size_y = h - (output_size_h - 1) * stride_y
            size_x = w - (output_size_w - 1) * stride_x

        log_debug3("\tstride_y {}", stride_y)
        log_debug3("\tstride_x {}", stride_x)
        log_debug3("\tsize_y {}", size_y)
        log_debug3("\tsize_x {}", size_x)


        ir_op = Pool2dOp(op_name,
                        pool_type=ir_graph.QNN_OP_POOL_AVG_2D,
                        size_x=size_x,
                        size_y=size_y,
                        stride_x=stride_x,
                        stride_y=stride_y)
        return ir_op


RelayTranslations.register_translation(RelayAdaptiveAvgPool2DTranslation(),
                                       converter_type('adaptive_avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   BatchMatMul
# ------------------------------------------------------------------------------
class RelayBatchMatMulTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBatchMatMulTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, MatMulOp.TRANSLATION_KEY, MatMulOp.LEGACY_TRANSLATION_KEY)
        if not quir_graph.has_buffer(input_names[0]):
            const_tensor = relay_params[input_names[0]]
            if isinstance(const_tensor, tvm.runtime.ndarray.NDArray) or isinstance(const_tensor, tvm.runtime.NDArray):
                const_tensor = const_tensor.asnumpy()
            const_op = ConstantOp(input_names[0], tensor=const_tensor)
            self.populate_quantization_params(relay_expr.args[0], converter_context, quir_graph, [input_names[0]], is_param=True)
            quir_graph.add(const_op, [], [input_names[0]], axis_formats=[AxisTracker.AxisFormat.ANY])
        if not quir_graph.has_buffer(input_names[1]):
            const_tensor = relay_params[input_names[1]]
            if isinstance(const_tensor, tvm.runtime.ndarray.NDArray) or isinstance(const_tensor, tvm.runtime.NDArray):
                const_tensor = const_tensor.asnumpy()
            const_op = ConstantOp(input_names[1], tensor=const_tensor)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [input_names[1]], is_param=True)
            quir_graph.add(const_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.ANY])
        shape_b = quir_graph.get_buffer(input_names[1]).shape
        # Our codebase relay does not support transpose yet
        # However, refer to tvm_src/src/relay/op/nn/nn.cc, BatchMatmulRel
        # input y is asserted to be transposed (line 939)
        ir_op = MatMulOp(op_name, transpose_in0=False, transpose_in1=True)
        return ir_op


RelayTranslations.register_translation(RelayBatchMatMulTranslation(),
                                       converter_type('batch_matmul', 'relay'))


# ------------------------------------------------------------------------------
#   BatchNorm
# ------------------------------------------------------------------------------
class RelayBatchNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBatchNormTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):

        attr_dict = {}
        batchnorm_attrs = relay_expr.attrs

        attr_dict["epsilon"] = batchnorm_attrs.epsilon if hasattr(batchnorm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = batchnorm_attrs.center if hasattr(batchnorm_attrs, 'center') else True
        attr_dict["scale"] = batchnorm_attrs.scale if hasattr(batchnorm_attrs, 'scale') else True
        attr_dict["axis"] = batchnorm_attrs.axis if hasattr(batchnorm_attrs, 'axis') else 1

        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        log_debug3("\taxis {}", attr_dict["axis"])

        return  attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, BatchnormOp.TRANSLATION_KEY,
                                                BatchnormOp.LEGACY_TRANSLATION_KEY)

        gamma = relay_params[input_names[1]].asnumpy()
        beta = relay_params[input_names[2]].asnumpy()
        moving_mean = relay_params[input_names[3]].asnumpy()
        moving_var = relay_params[input_names[4]].asnumpy()

        log_debug3("\tgamma shape {}", gamma.shape)
        log_debug3("\tbeta shape {}", beta.shape)
        log_debug3("\tmoving_mean shape {}", moving_mean.shape)
        log_debug3("\tmoving_var shape {}", moving_var.shape)

        if attr_dict["axis"] != 3:
            raise ValueError("In NHWC data layout, batchnorm channel is dimension 3, got {}".format(attr_dict["axis"]))

        center = attr_dict["center"]
        scale = attr_dict["scale"]
        epsilon = attr_dict["epsilon"]

        self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [input_names[1]], is_param=True)
        self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
        gamma_quant_enc = quir_graph.get_overridden_encoding(input_names[1])
        beta_quant_enc = quir_graph.get_overridden_encoding(input_names[2])
        if gamma_quant_enc:
            quantized_gamma = translation_utils.quantize_params(gamma, gamma_quant_enc[0])
            gamma = translation_utils.dequantize_params(quantized_gamma, gamma_quant_enc[0])
            # remove gamma encodings since already applied
            quir_graph.remove_overridden_encoding(input_names[1])
        if beta_quant_enc:
            quantized_beta = translation_utils.quantize_params(beta, beta_quant_enc[0])
            beta = translation_utils.dequantize_params(quantized_beta, beta_quant_enc[0])
            # remove beta encodings since already applied
            quir_graph.remove_overridden_encoding(input_names[2])

        # weights = gamma/sqrt(var+epsilon)
        weights = gamma / np.sqrt(moving_var + epsilon)
        # bias = -mu/sqrt(var+epsilon)
        bias = -moving_mean / np.sqrt(moving_var + epsilon)
        if scale:
            # bias = -mu*gamma/sqrt(var+epsilon)
            bias *= gamma
        if center:
            # bias = -mu/sqrt(var+epsilon) + beta or bias = -mu*gamma/sqrt(var+epsilon) + beta
            bias += beta

        weights_name = op_name + "_bn_w"
        bias_name = op_name + "_bn_b"
        weights_constant_op = ConstantOp(weights_name, tensor=weights)
        bias_constant_op = ConstantOp(bias_name, tensor=bias)
        weight_node = quir_graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        bias_node = quir_graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        quir_graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
        quir_graph.add_src_op_info(bias_name, None, bias_node.output_names[0])

        ir_op = BatchnormOp(op_name)

        for name in input_names[1:]:
            input_names.remove(name)
        input_names.append(weights_name)
        input_names.append(bias_name)
        return ir_op


RelayTranslations.register_translation(RelayBatchNormTranslation(),
                                       converter_type('batch_norm', 'relay'))


# ------------------------------------------------------------------------------
#   BiasAdd
# ------------------------------------------------------------------------------
class RelayBiasaddTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayBiasaddTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, ir_graph.QNN_OP_ELEMENT_WISE_ADD,
                                                ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD])

        bias = relay_params[input_names[1]]
        if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
            bias = bias.asnumpy().astype(np.float32)

        log_debug3("\tbias shape {}", bias.shape)

        bias_name = op_name + "_const_bias"
        bias_name = validate_const_name(quir_graph, input_names[1], bias_name)
        input_names[1] = bias_name
        if not quir_graph.has_buffer(bias_name):
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [bias_name], is_param=True)
            quir_graph.add(ConstantOp(bias_name, bias), [], [bias_name])

        ir_op = ElementwiseBinaryOp(op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)

        return ir_op


RelayTranslations.register_translation(RelayBiasaddTranslation(),
                                       converter_type('bias_add', 'relay'))


# ------------------------------------------------------------------------------
#   ChannelShuffle
# ------------------------------------------------------------------------------
class RelayChannelShuffleTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayChannelShuffleTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["groups"] = int(relay_expr.attrs.groups)
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ChannelShuffleOp.TRANSLATION_KEY,
                                                ChannelShuffleOp.LEGACY_TRANSLATION_KEY)
        return ChannelShuffleOp(op_name, num_groups=attr_dict["groups"])

RelayTranslations.register_translation(RelayChannelShuffleTranslation(),
                                       converter_type("channel_shuffle", "relay"))


# ------------------------------------------------------------------------------
#   Conv Base
# ------------------------------------------------------------------------------
class RelayConvBaseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConvBaseTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        conv_attrs = relay_expr.attrs

        attr_dict["kernel_layout"] = conv_attrs.kernel_layout
        log_debug3("\tkernel_layout {}", conv_attrs.kernel_layout)

        padding = [int(val) for val in conv_attrs.padding]
        log_debug3("\tpadding {}", padding)
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_left = padding[1]
            pad_bottom = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pad_top"] = pad_top
        attr_dict["pad_bottom"] = pad_bottom
        attr_dict["pad_left"] = pad_left
        attr_dict["pad_right"] = pad_right

        attr_dict["padding_size_strategy"] = ir_graph.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        strides = [int(val) for val in conv_attrs.strides]
        log_debug3("\tstrides {}", strides)
        # y -> height
        # x -> width
        stride_y = strides[0]
        stride_x = strides[1]
        attr_dict["stride_x"] = stride_x
        attr_dict["stride_y"] = stride_y

        dilation = [int(val) for val in conv_attrs.dilation]
        log_debug3("\tdilation {}", dilation)
        # y -> height
        # x -> width
        dilation_y = dilation[0]
        dilation_x = dilation[1]
        attr_dict["dilation_x"] = dilation_x
        attr_dict["dilation_y"] = dilation_y

        groups = int(conv_attrs.groups)
        log_debug3("\tgroups {}", groups)
        attr_dict["groups"] = groups

        return attr_dict


# ------------------------------------------------------------------------------
#   Conv
# ------------------------------------------------------------------------------
class RelayConvTranslation(RelayConvBaseTranslation):
    def __init__(self):
        super(RelayConvTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        if input_names[1] not in relay_params:
            raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[1]))

        weights = relay_params[input_names[1]]
        if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
            weights = weights.asnumpy()
        log_debug3("\tweights shape {}", weights.shape)

        kernel_layout = attr_dict["kernel_layout"]
        if kernel_layout == "HWIO":
            pass
        elif kernel_layout == "HWOI":
            log_debug3("\tHWOI kernel layout with shape {} detected, "
                       "Transposing the weights to make it 'HWIO'.".format(weights.shape))
            weights = np.transpose(weights, AxisTracker.AxisFormat.HWOI_TO_HWIO)
            weights = np.ascontiguousarray(weights)
            kernel_layout = "HWIO"
            log_debug3("\tTransposed weights to be of shape {}", weights.shape)
        else:
            raise ValueError("Unsupported kernel layout {}".format(kernel_layout))

        # Handle marking this Convolution as a DepthwiseConvolution
        num_input_channels = quir_graph.src_axis_order.extract_2d_spatial_dims(
            quir_graph.get_buffer(input_names[0]).shape)[-1]
        num_output_channels = weights.shape[kernel_layout.find('O')]
        convolution_class = Conv2dOp
        if attr_dict["groups"] == num_input_channels and num_input_channels == num_output_channels:
            convolution_class = DepthwiseConv2dOp
            log_debug3("\tReshaping depthwise convolution weights of shape {}", weights.shape)
            weights = np.reshape(weights, (weights.shape[0], weights.shape[1], 1, -1))
            log_debug3("\tReshaped depthwise convolution weights to shape {}", weights.shape)

        op_name = converter_context.get_op_name(relay_expr, convolution_class.TRANSLATION_KEY,
                                                convolution_class.LEGACY_TRANSLATION_KEY)

        weight_name = op_name + "_const_weight"
        weight_name = validate_const_name(quir_graph, input_names[1], weight_name)
        input_names[1] = weight_name

        if not quir_graph.has_buffer(weight_name):
            weights_op = ConstantOp(weight_name, tensor=weights)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [weight_name], is_param=True)
            quir_graph.add(weights_op, [], [weight_name], axis_formats=[AxisTracker.AxisFormat.HWIO])

        if len(input_names) > 2:
            if input_names[2] not in relay_params:
                raise ValueError("Unsupported dynamic biases on tensor {}".format(input_names[2]))
            bias = relay_params[input_names[2]]
            if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
                bias = bias.asnumpy().astype(np.float32)
            log_debug3("\tbias shape {}", bias.shape)
            bias_op = ConstantOp(input_names[2], tensor=bias)
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
            quir_graph.add(bias_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        ir_op = convolution_class(op_name,
                                  padx_before=attr_dict["pad_left"],
                                  padx_after=attr_dict["pad_right"],
                                  pady_before=attr_dict["pad_top"],
                                  pady_after=attr_dict["pad_bottom"],
                                  stridex=attr_dict["stride_x"],
                                  stridey=attr_dict["stride_y"],
                                  dilationx=attr_dict["dilation_x"],
                                  dilationy=attr_dict["dilation_y"],
                                  groups=attr_dict["groups"],
                                  padding_size_strategy=attr_dict["padding_size_strategy"])

        return ir_op


RelayTranslations.register_translation(RelayConvTranslation(),
                                       converter_type('conv2d', 'relay'))


# ------------------------------------------------------------------------------
#   Conv2D_Transpose
# ------------------------------------------------------------------------------
class RelayConvTransposeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConvTransposeTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        conv_attrs = relay_expr.attrs

        log_debug3("\tdata layout {}", conv_attrs.data_layout)
        if conv_attrs.data_layout != "NHWC":
            # QUIR expects data to be "NHWC"
            raise ValueError("Unsupported data layout {}".format(conv_attrs.data_layout))

        log_debug3("\tkernel layout {}", conv_attrs.kernel_layout)
        if conv_attrs.kernel_layout != "OIHW":
            raise ValueError("Unsupported kernel layout {}".format(conv_attrs.kernel_layout))
        attr_dict["kernel_layout"] = conv_attrs.kernel_layout

        log_debug3("\tout layout {}", conv_attrs.out_layout)
        if conv_attrs.out_layout != "":
            # This attribute is not supported, so only empty/default is accepted
            raise ValueError("Unsupported out layout {}".format(conv_attrs.out_layout))

        log_debug3("\tout dtype {}", conv_attrs.out_dtype)
        if conv_attrs.out_dtype != "float32":
            # Only float32 is currently supported
            raise ValueError("Unsupported out dtype {}".format(conv_attrs.out_dtype))

        padding = [int(val) for val in conv_attrs.padding]
        log_debug3("\tpadding {}", padding)
        if len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            pad_top = padding[0]
            pad_left = padding[1]
            pad_bottom = padding[2]
            pad_right = padding[3]
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pad_top"] = pad_top
        attr_dict["pad_bottom"] = pad_bottom
        attr_dict["pad_left"] = pad_left
        attr_dict["pad_right"] = pad_right

        attr_dict["padding_size_strategy"] = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        strides = [int(val) for val in conv_attrs.strides]
        log_debug3("\tstrides {}", strides)
        # y -> height
        # x -> width
        stride_y = strides[0]
        stride_x = strides[1]
        attr_dict["stride_x"] = stride_x
        attr_dict["stride_y"] = stride_y

        dilation = [int(val) for val in conv_attrs.dilation]
        log_debug3("\tdilation {}", dilation)
        # y -> height
        # x -> width
        dilation_y = dilation[0]
        dilation_x = dilation[1]
        attr_dict["dilation_x"] = dilation_x
        attr_dict["dilation_y"] = dilation_y

        groups = int(conv_attrs.groups)
        log_debug3("\tgroups {}", groups)
        attr_dict["groups"] = groups

        output_padding = conv_attrs.output_padding
        log_debug3("\toutput padding {}", conv_attrs.output_padding)
        # FIXME: This attribute can have 1, 2, or 4 numbers.
        # refer to tvm_src/include/tvm/relay/attrs/nn.h, Conv2DTransposeAttrs.
        attr_dict["output_padding_y"] = output_padding[0]
        attr_dict["output_padding_x"] = output_padding[1]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TransposeConv2dOp.TRANSLATION_KEY,
                                                TransposeConv2dOp.LEGACY_TRANSLATION_KEY)

        kernel_layout = attr_dict["kernel_layout"]
        if kernel_layout == "OIHW":
            if input_names[1] not in relay_params:
                raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[1]))
            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()
            log_debug3("\tweights shape {}", weights.shape)
            weights = np.transpose(weights, AxisTracker.AxisFormat.OIHW_TO_HWOI)
            weights = np.ascontiguousarray(weights)
            log_debug3("\ttransposed deconv weights to {}", weights.shape)
        else:
            raise ValueError("Unsupported kernel layout {}".format(kernel_layout))

        weights_op = ConstantOp(input_names[1], tensor=weights)
        self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [weights_op.name], is_param=True)
        quir_graph.add(weights_op, [], [input_names[1]], axis_formats=[AxisTracker.AxisFormat.HWIO])

        if len(input_names) > 2:
            if input_names[2] not in relay_params:
                raise ValueError("Unsupported dynamic biases on tensor {}".format(input_names[2]))
            bias = relay_params[input_names[2]]
            if isinstance(bias, tvm.runtime.ndarray.NDArray) or isinstance(bias, tvm.runtime.NDArray):
                bias = bias.asnumpy().astype(np.float32)
            log_debug3("\tbias shape {}", bias.shape)
            bias_op = ConstantOp(input_names[2], tensor=bias)
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
            quir_graph.add(bias_op, [], [input_names[2]], axis_formats=[AxisTracker.AxisFormat.ANY])

        ir_op = TransposeConv2dOp(op_name,
                                  padx_before=attr_dict["pad_left"],
                                  padx_after=attr_dict["pad_right"],
                                  pady_before=attr_dict["pad_top"],
                                  pady_after=attr_dict["pad_bottom"],
                                  stridex=attr_dict["stride_x"],
                                  stridey=attr_dict["stride_y"],
                                  dilationx=attr_dict["dilation_x"],
                                  dilationy=attr_dict["dilation_y"],
                                  output_paddingx=attr_dict["output_padding_x"],
                                  output_paddingy=attr_dict["output_padding_y"],
                                  groups=attr_dict["groups"],
                                  padding_size_strategy=attr_dict["padding_size_strategy"])

        return ir_op


RelayTranslations.register_translation(RelayConvTransposeTranslation(),
                                       converter_type('conv2d_transpose', 'relay'))


# ------------------------------------------------------------------------------
#   Dense
# ------------------------------------------------------------------------------
class RelayDenseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDenseTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        if input_names[1] in relay_params:
            op_name = converter_context.get_op_name(relay_expr, FullyConnectedOp.TRANSLATION_KEY,
                                                    FullyConnectedOp.LEGACY_TRANSLATION_KEY)

            weights = relay_params[input_names[1]]
            if isinstance(weights, tvm.runtime.ndarray.NDArray) or isinstance(weights, tvm.runtime.NDArray):
                weights = weights.asnumpy()

            # Weights has shape [out_units, in_units]
            weights_constant_op = ConstantOp(input_names[1], tensor=weights)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [input_names[1]], is_param=True)
            weights_node = quir_graph.add(weights_constant_op, [], [input_names[1]])
            quir_graph.add_src_op_info(input_names[1], None, weights_node.output_names[0])

            bias = np.zeros(weights.shape[-2], dtype=np.float32)
            bias_name = op_name + "_fc_b"
            bias_constant_op = ConstantOp(bias_name, tensor=bias)
            bias_node = quir_graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(bias_name, None, bias_node.output_names[0])
            log_debug3("\tweight shape {}", weights.shape)
            log_debug3("\tbias shape {}", bias.shape)

            ir_op = FullyConnectedOp(op_name)
            input_names.append(bias_name)
        else:
            op_name = converter_context.get_op_name(relay_expr, MatMulOp.TRANSLATION_KEY,
                                                    MatMulOp.LEGACY_TRANSLATION_KEY)
            ir_op = MatMulOp(op_name, transpose_in0=False, transpose_in1=True)

        return ir_op


RelayTranslations.register_translation(RelayDenseTranslation(),
                                       converter_type('dense', 'relay'))


# ------------------------------------------------------------------------------
#   DepthToSpace
# ------------------------------------------------------------------------------
class RelayDepthToSpaceTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDepthToSpaceTranslation, self).__init__()
        self.SUPPORTED_DEPTHTOSPACE_MODES = {'DCR': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR,
                                             'CRD': ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD}

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        dts_attrs = relay_expr.attrs

        attr_dict["layout"] = dts_attrs.layout
        attr_dict["mode"] = dts_attrs.mode
        attr_dict["block_size"] = dts_attrs.block_size
        log_debug3("\tDepthToSpaceOp data layout {}, mode {}, block size {}", dts_attrs.layout, dts_attrs.mode, dts_attrs.block_size)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, DepthToSpaceOp.TRANSLATION_KEY,
                                                DepthToSpaceOp.LEGACY_TRANSLATION_KEY)

        log_assert(attr_dict["mode"] in self.SUPPORTED_DEPTHTOSPACE_MODES,
                   "DepthToSpace only support DCR and CRD mode, but got {}", attr_dict["mode"])
        log_assert(attr_dict["layout"] == "NHWC",
                   "DepthToSpace only support NHWC data layout, but got {}", attr_dict["layout"])

        block_size = [attr_dict["block_size"]] * 2
        mode = self.SUPPORTED_DEPTHTOSPACE_MODES[attr_dict["mode"]]

        ir_op = DepthToSpaceOp(op_name,
                               block_size=block_size,
                               mode=mode)

        return ir_op


RelayTranslations.register_translation(RelayDepthToSpaceTranslation(),
                                       converter_type('depth_to_space', 'relay'))


# ------------------------------------------------------------------------------
#   Detecion PostPorcess
# ------------------------------------------------------------------------------
class RelayDetectionPostProcessTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDetectionPostProcessTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = dict(relay_expr.attrs)
        attr_dict['use_bg_in_nms'] = False if attr_dict['use_bg_in_nms'] == 0 else True
        attr_dict['output_background'] =  False if attr_dict['output_background'] == 0 else True
        attr_dict['share_location'] =  False if attr_dict['share_location'] == 0 else True

        log_debug3("\tuse_bg_in_nms {}", attr_dict['use_bg_in_nms'])
        log_debug3("\toutput_background {}", attr_dict['output_background'])
        log_debug3("\tshare_location {}", attr_dict['share_location'])

        def get_prim_type(v):
            if isinstance(v, tvm.tir.expr.IntImm):
                return v.value
            elif isinstance(v, tvm.tir.expr.FloatImm):
                return v.value
            elif isinstance(v, tvm.ir.container.Array):
                return [get_prim_type(i) for i in list(v)]
            elif isinstance(v, tvm.runtime.container.String):
                return str(v)
            else:
                return v

        for k, v in attr_dict.items():
            attr_dict[k] = get_prim_type(v)
            log_debug3("\t{} {}", k, v)

        return attr_dict


    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, DetectionOutputOp.TRANSLATION_KEY,
                                                DetectionOutputOp.LEGACY_TRANSLATION_KEY)
        if input_names[2] not in relay_params:
            raise ValueError("Unsupported dynamic weights on tensor {}".format(input_names[2]))
        self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [input_names[2]], is_param=True)
        quir_graph.add(ConstantOp(input_names[2], relay_params[input_names[2]].asnumpy()), [], [input_names[2]])

        ir_op = DetectionOutputOp(op_name,
                                  output_dims=attr_dict['output_dims'],
                                  delta_scaling_factors=attr_dict['delta_scaling_factors'],
                                  confidence_threshold=attr_dict['confidence_threshold'],
                                  iou_threshold=attr_dict['iou_threshold'],
                                  nms_type=attr_dict['nms_type'],
                                  background_class_idx=attr_dict['background_class_idx'],
                                  use_bg_in_nms=attr_dict['use_bg_in_nms'],
                                  output_background=attr_dict['output_background'],
                                  share_location=attr_dict['share_location'],
                                  nms_eta=attr_dict['nms_eta'],
                                  detection_limit=attr_dict['detection_limit'])

        return ir_op


RelayTranslations.register_translation(RelayDetectionPostProcessTranslation(),
                                       converter_type('detection_postprocess', 'relay'))


# ------------------------------------------------------------------------------
#   Dropout
# ------------------------------------------------------------------------------
class RelayDropoutTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDropoutTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, IdentityOp.TRANSLATION_KEY,
                                                IdentityOp.LEGACY_TRANSLATION_KEY)
        return IdentityOp(op_name)


RelayTranslations.register_translation(RelayDropoutTranslation(),
                                       converter_type('dropout', 'relay'))


# ------------------------------------------------------------------------------
#   Global Average Pool2D
# ------------------------------------------------------------------------------
class RelayGlobalAvgPool2DTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGlobalAvgPool2DTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['layout'] = relay_expr.attrs.layout

        log_debug3("\tlayout {}", attr_dict['layout'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, Pool2dOp.TRANSLATION_KEY, Pool2dOp.LEGACY_TRANSLATION_KEY)
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        layout = attr_dict['layout']

        if layout == "NHWC":
            size_x = input_shape[2]
            size_y = input_shape[1]
        else:
            raise ValueError("No support {} data layout".format(layout))

        log_debug3("\tsize_x {}", size_x)
        log_debug3("\tsize_y {}", size_y)

        ir_op = Pool2dOp(op_name,
                        pool_type=ir_graph.QNN_OP_POOL_AVG_2D,
                        size_x=size_x,
                        size_y=size_y)
        return ir_op


RelayTranslations.register_translation(RelayGlobalAvgPool2DTranslation(),
                                       converter_type('global_avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   LayerNorm
# ------------------------------------------------------------------------------
class RelayLayerNormTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLayerNormTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        layernorm_attrs = relay_expr.attrs
        attr_dict["axis"] = layernorm_attrs.axis if hasattr(layernorm_attrs, 'axis') else -1
        attr_dict["epsilon"] = layernorm_attrs.epsilon if hasattr(layernorm_attrs, 'epsilon') else 1e-5
        attr_dict["center"] = layernorm_attrs.center if hasattr(layernorm_attrs, 'center') else True
        attr_dict["scale"] = layernorm_attrs.scale if hasattr(layernorm_attrs, 'scale') else True
        log_debug3("\taxis {}", attr_dict["axis"])
        log_debug3("\tepsilon {}", attr_dict["epsilon"])
        log_debug3("\tcenter {}", attr_dict["center"])
        log_debug3("\tscale {}", attr_dict["scale"])
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, LayerNormOp.TRANSLATION_KEY, LayerNormOp.LEGACY_TRANSLATION_KEY)
        if attr_dict["scale"]:
            gamma = relay_params[input_names[1]].asnumpy()
            log_debug3("\tgamma shape {}", gamma.shape)
            gamma_name = op_name + "_gamma"
            gamma_constant_op = ConstantOp(gamma_name, tensor=gamma)
            self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [gamma_name], is_param=True)
            gamma_node = quir_graph.add(gamma_constant_op, [], [gamma_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(gamma_name, None, gamma_node.output_names[0])
        if attr_dict["center"]:
            beta = relay_params[input_names[2]].asnumpy()
            log_debug3("\tbeta shape {}", beta.shape)
            beta_name = op_name + "_beta"
            beta_constant_op = ConstantOp(beta_name, tensor=beta)
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [beta_name], is_param=True)
            beta_node = quir_graph.add(beta_constant_op, [], [beta_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            quir_graph.add_src_op_info(beta_name, None, beta_node.output_names[0])
        ir_op = LayerNormOp(op_name,
                            epsilon=attr_dict["epsilon"],
                            axes=[attr_dict["axis"]])
        # update input names
        for name in input_names[1:]:
            input_names.remove(name)
        if attr_dict["scale"]:
            input_names.append(gamma_name)
        if attr_dict["center"]:
            input_names.append(beta_name)
        return ir_op


RelayTranslations.register_translation(RelayLayerNormTranslation(),
                                       converter_type('layer_norm', 'relay'))


# ------------------------------------------------------------------------------
#   LeakyRelu
# ------------------------------------------------------------------------------
class RelayLeakyReluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLeakyReluTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        leaky_relu_attrs = relay_expr.attrs
        attr_dict["alpha"] = leaky_relu_attrs.alpha
        log_debug3("\talpha {}", attr_dict["alpha"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, PreluOp.TRANSLATION_KEY, PreluOp.LEGACY_TRANSLATION_KEY)

        alpha = attr_dict["alpha"]
        coeff = alpha * np.ones(quir_graph.get_buffer(input_names[0]).shape[-1], dtype=np.float32)

        coeff_name = op_name + "_coeff"
        coeff_constant_op = ConstantOp(coeff_name, tensor=coeff)
        coeff_node = quir_graph.add(coeff_constant_op, [], [coeff_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        quir_graph.add_src_op_info(coeff_name, None, coeff_node.output_names[0])
        ir_op = PreluOp(op_name)
        input_names.append(coeff_name)
        return ir_op


RelayTranslations.register_translation(RelayLeakyReluTranslation(),
                                       converter_type('leaky_relu', 'relay'))


# ------------------------------------------------------------------------------
#   LogSoftmax
# ------------------------------------------------------------------------------
class RelayLogSoftmaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLogSoftmaxTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, LogSoftmaxOp.TRANSLATION_KEY, LogSoftmaxOp.LEGACY_TRANSLATION_KEY)

        ir_op = LogSoftmaxOp(op_name, axis=attr_dict["axis"])
        return ir_op


# ------------------------------------------------------------------------------
#   PadOp
# ------------------------------------------------------------------------------
class RelayPadTranslation(RelayTranslationBase):
    class RelayPadMode:
        CONSTANT = 'constant'
        REFLECT = 'reflect'
        EDGE = 'edge'
    def __init__(self):
        super(RelayPadTranslation, self).__init__()
        self.supported_modes = {self.RelayPadMode.CONSTANT : ir_graph.QNN_OP_PAD_SCHEME_CONSTANT,
                                self.RelayPadMode.REFLECT : ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT,
                                self.RelayPadMode.EDGE : ir_graph.QNN_OP_PAD_SCHEME_EDGE}

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        pad_pairs = list()
        for pad in relay_expr.attrs.pad_width:
            pad_pairs.append([int(i) for i in pad])

        attr_dict["pad_pairs"] = pad_pairs
        attr_dict["pad_mode"] = relay_expr.attrs.pad_mode

        # pad value from float, or tvm.relay.Expr, optional, default=0
        # if not in relay_expr.attrs, it will be default value or tvm.relay.Expr
        if hasattr(relay_expr.attrs, 'pad_value'):
            attr_dict["pad_value"] = relay_expr.attrs.pad_value
        else:
            attr_dict["pad_value"] = None

        log_debug3("\tpad_pairs {}", pad_pairs)
        log_debug3("\tpad_mode {}", attr_dict["pad_mode"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PadOp.TRANSLATION_KEY, PadOp.LEGACY_TRANSLATION_KEY)

        pad_pairs = attr_dict["pad_pairs"]
        pad_pairs = np.asarray(pad_pairs, dtype=np.dtype('int32'))
        mode = attr_dict["pad_mode"]
        pad_value = attr_dict["pad_value"]

        if pad_value is None:
            # pad constant value from inputs[1] expr.Constant
            # if no found constant from param, set to zero by default
            pad_value_op_name = input_names[1]
            if pad_value_op_name in relay_params:
                expr_const_pad_value = relay_params[pad_value_op_name]
                pad_value = float(expr_const_pad_value.asnumpy())
            else:
                log_debug2("\tNo Padding value, use default as zero")

                pad_value = 0

        log_debug3("\tpad_value {}", pad_value)

        ir_op = PadOp(op_name,
                        pad_amount=pad_pairs,
                        pad_constant_value=pad_value,
                        scheme=self.supported_modes[mode])

        # Only data input is needed in IR graph. Pad value input is ignored
        for name in input_names[1:]:
            input_names.remove(name)

        return ir_op


RelayTranslations.register_translation(RelayPadTranslation(),
                                       converter_type('pad', 'relay'))


# ------------------------------------------------------------------------------
#   Pooling Base
# ------------------------------------------------------------------------------
class RelayPoolingBaseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayPoolingBaseTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        pool_attrs = relay_expr.attrs

        if pool_attrs.layout != "NHWC":
            raise ValueError("Unsupported layout {}".format(pool_attrs.layout))

        pool_size = pool_attrs.pool_size
        log_debug3("\tpool_size {}", pool_size)
        if isinstance(pool_size, int):
            attr_dict["size_x"] = attr_dict["size_y"] = int(pool_size)
        else:
            # y -> height
            # x -> width
            attr_dict["size_y"] = int(pool_size[0])
            attr_dict["size_x"] = int(pool_size[1])

        padding = pool_attrs.padding
        log_debug3("\tpadding {}", padding)
        if len(padding) == 2:
            pad_top = pad_bottom = int(padding[0])
            pad_left = pad_right = int(padding[1])
        elif len(padding) == 4:
            pad_top = int(padding[0])
            pad_left = int(padding[1])
            pad_bottom = int(padding[2])
            pad_right = int(padding[3])
        else:
            raise ValueError("Unsupported Padding value {}".format(padding))
        attr_dict["pady_before"] = pad_top
        attr_dict["pady_after"] = pad_bottom
        attr_dict["padx_before"] = pad_left
        attr_dict["padx_after"] = pad_right

        strides = [int(val) for val in pool_attrs.strides]
        log_debug3("\tstrides {}", strides)
        # y -> height
        # x -> width
        stride_y = strides[0]
        stride_x = strides[1]
        attr_dict["stride_x"] = int(stride_x)
        attr_dict["stride_y"] = int(stride_y)

        ceil_mode = getattr(pool_attrs, "ceil_mode", False)
        if ceil_mode:
            attr_dict["padding_size_strategy"] = ir_graph.PADDING_SIZE_EXPLICIT
        else:
            attr_dict["padding_size_strategy"] = ir_graph.PADDING_SIZE_EXPLICIT_FLOOR
        log_debug3("\tpadding strategy {}", attr_dict["padding_size_strategy"])

        attr_dict["count_pad_for_edges"] = getattr(pool_attrs, "count_include_pad", False)
        log_debug3("\count_pad_for_edges {}", attr_dict["count_pad_for_edges"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, Pool2dOp.TRANSLATION_KEY, Pool2dOp.LEGACY_TRANSLATION_KEY)

        ir_op = Pool2dOp(op_name,
                        pool_type=attr_dict["pool_type"],
                        size_x=attr_dict["size_x"],
                        size_y=attr_dict["size_y"],
                        stride_x=attr_dict["stride_x"],
                        stride_y=attr_dict["stride_y"],
                        padx_before=attr_dict["padx_before"],
                        padx_after=attr_dict["padx_after"],
                        pady_before=attr_dict["pady_before"],
                        pady_after=attr_dict["pady_after"],
                        padding_size_strategy=attr_dict["padding_size_strategy"],
                        count_pad_for_edges=attr_dict["count_pad_for_edges"])

        return ir_op


# ------------------------------------------------------------------------------
#   AvgPooling2D
# ------------------------------------------------------------------------------
class RelayAvgPoolTranslation(RelayPoolingBaseTranslation):
    def __init__(self):
        super(RelayAvgPoolTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = super().extract_attributes(relay_expr, relay_params)
        attr_dict["pool_type"] = ir_graph.QNN_OP_POOL_AVG_2D
        return attr_dict


RelayTranslations.register_translation(RelayAvgPoolTranslation(),
                                       converter_type('avg_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   MaxPooling2D
# ------------------------------------------------------------------------------
class RelayMaxPoolTranslation(RelayPoolingBaseTranslation):
    def __init__(self):
        super(RelayMaxPoolTranslation, self).__init__()

    # Returns a dictionary of parameters
    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = super().extract_attributes(relay_expr, relay_params)
        attr_dict["pool_type"] = ir_graph.QNN_OP_POOL_MAX_2D
        return attr_dict


RelayTranslations.register_translation(RelayMaxPoolTranslation(),
                                       converter_type('max_pool2d', 'relay'))


# ------------------------------------------------------------------------------
#   MirrorPadOp
# ------------------------------------------------------------------------------
class RelayMirrorPadTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayMirrorPadTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["mode"] = relay_expr.attrs.mode
        attr_dict["pad_width"] = relay_expr.attrs.pad_width
        log_debug3("\tmode, {}, pad_width {}", attr_dict["mode"], attr_dict["pad_width"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PadOp.TRANSLATION_KEY, PadOp.LEGACY_TRANSLATION_KEY)

        mode = attr_dict["mode"]
        pad_width = attr_dict["pad_width"]

        # The type of pad_width affect the type of output shape,
        # so we translate it to native Python types.
        ir_pad_width = []
        for pad_per_axis in pad_width:
            ir_pad_width.append([int(pad_per_axis[0]), int(pad_per_axis[1])])

        if mode == "SYMMETRIC":
            ir_mode = ir_graph.QNN_OP_PAD_SCHEME_MIRROR_SYMMETRIC
        elif mode == "REFLECT":
            ir_mode = ir_graph.QNN_OP_PAD_SCHEME_MIRROR_REFLECT
        else:
            log_assert(False, "Unknown nn.mirror_pad mode: {}", mode)

        ir_op = PadOp(op_name,
                pad_amount=ir_pad_width,
                scheme=ir_mode)

        return ir_op


RelayTranslations.register_translation(RelayMirrorPadTranslation(),
                                       converter_type('mirror_pad', 'relay'))


# ------------------------------------------------------------------------------
#   Prelu
# ------------------------------------------------------------------------------
class RelayPreluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayPreluTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        prelu_attrs = relay_expr.attrs
        attr_dict["axis"] = prelu_attrs.axis
        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, PreluOp.TRANSLATION_KEY, PreluOp.LEGACY_TRANSLATION_KEY)

        channel_axis = attr_dict["axis"]
        slope_input_name = input_names[1]

        input_shape = quir_graph.get_buffer(input_names[0]).shape

        log_assert(channel_axis == len(input_shape)-1,
                   "Expect the channel axis is the last dimension, but got "
                   "channel_axis={} for data_tensor_rank={}",
                   channel_axis, len(input_shape))

        log_assert(slope_input_name in relay_params,
                   "Only support PRelu with constant slope(second input). "
                   "But {} is not in relay_params.",
                   slope_input_name)

        slope = relay_params[slope_input_name]
        if isinstance(slope, (tvm.runtime.ndarray.NDArray, tvm.runtime.NDArray)):
            slope = slope.asnumpy().astype(np.float32)

        coeff_name = slope_input_name
        coeff_constant_op = ConstantOp(coeff_name, tensor=slope)
        self.populate_quantization_params(relay_expr.args[1], converter_context, quir_graph, [coeff_name], is_param=True)
        coeff_node = quir_graph.add(coeff_constant_op, [], [coeff_name], axis_formats=[AxisTracker.AxisFormat.ANY])
        quir_graph.add_src_op_info(coeff_name, None, coeff_node.output_names[0])
        ir_op = PreluOp(op_name)
        return ir_op


RelayTranslations.register_translation(RelayPreluTranslation(),
                                       converter_type('prelu', 'relay'))


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class RelayReluTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReluTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY, NeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = NeuronOp(op_name, ir_graph.QNN_OP_RELU)
        return ir_op


RelayTranslations.register_translation(RelayReluTranslation(),
                                       converter_type('relu', 'relay'))


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class RelaySigmoidTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySigmoidTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY, NeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = NeuronOp(op_name,
                         ir_graph.QNN_OP_SIGMOID,
                         alpha=1.0)
        return ir_op


RelayTranslations.register_translation(RelaySigmoidTranslation(),
                                       converter_type('sigmoid', 'relay'))


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class RelaySoftmaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySoftmaxTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SoftmaxOp.TRANSLATION_KEY, SoftmaxOp.LEGACY_TRANSLATION_KEY)

        ir_op = SoftmaxOp(op_name, axis=attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelaySoftmaxTranslation(),
                                       converter_type('softmax', 'relay'))


# ------------------------------------------------------------------------------
#   SpaceToDepth
# ------------------------------------------------------------------------------
class RelaySpaceToDepthTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySpaceToDepthTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["block_size"] = relay_expr.attrs.block_size
        attr_dict["layout"] = relay_expr.attrs.layout
        attr_dict["mode"] = relay_expr.attrs.mode

        log_debug3("\tblock_size {}, layout {}, mode {}",
                   attr_dict["block_size"], attr_dict["layout"], attr_dict["mode"])


        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SpaceToDepthOp.TRANSLATION_KEY,
                                                SpaceToDepthOp.LEGACY_TRANSLATION_KEY)

        log_assert(not attr_dict["mode"] or attr_dict["mode"] == "DCR",
                   "SpaceToDepth only support DCR mode, but got {}", attr_dict["mode"])
        log_assert(attr_dict["layout"] == "NHWC",
                   "SpaceToDepth only support NHWC layout, but got {}", attr_dict["layout"])

        block_size = [attr_dict["block_size"]] * 2

        ir_op = SpaceToDepthOp(op_name, block_size=block_size)
        return ir_op


RelayTranslations.register_translation(RelaySpaceToDepthTranslation(),
                                       converter_type('space_to_depth', 'relay'))


# ------------------------------------------------------------------------------
#   Upsampling
# ------------------------------------------------------------------------------
class RelayUpsamplingTranslation(RelayTranslationBase):

    # scaling method names in relay
    class ScaleModes:
        BICUBIC = "bicubic"
        BILINEAR = "bilinear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    # name mapping from relay to quir
    RELAY_CONSTS_TO_IR = {
        ScaleModes.BILINEAR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        ScaleModes.NEAREST_NEIGHBOR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST
    }

    def __init__(self):
        super(RelayUpsamplingTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        upsampling_attrs = relay_expr.attrs
        attr_dict["scale_h"] = getattr(upsampling_attrs, "scale_h")
        attr_dict["scale_w"] = getattr(upsampling_attrs, "scale_w")
        log_debug3("\tscale_h {}", attr_dict["scale_h"])
        log_debug3("\tscale_w {}", attr_dict["scale_w"])

        attr_dict["layout"] = getattr(upsampling_attrs, "layout")
        log_debug3("\tlayout {}", attr_dict["layout"])

        scale_mode = getattr(upsampling_attrs, "method", self.ScaleModes.NEAREST_NEIGHBOR)
        if scale_mode == self.ScaleModes.BICUBIC:
            raise ValueError("Unsupported scale method {}".format(scale_mode))

        attr_dict["interpolation_mode"] = self.RELAY_CONSTS_TO_IR[scale_mode]
        log_debug3("\tinterpolation_mode mode {}", attr_dict["interpolation_mode"])

        transform_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC
        align_corners = getattr(upsampling_attrs, "align_corners", False)
        if align_corners:
            transform_mode = ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS
        attr_dict["transformation_mode"] = transform_mode
        log_debug3("\ttransformation_mode mode {}", attr_dict['transformation_mode'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ResizeOp.TRANSLATION_KEY,
                                                ResizeOp.LEGACY_TRANSLATION_KEY)
        if attr_dict["layout"] != "NHWC":
            raise ValueError("Unsupported data layout {}".format(attr_dict["layout"]))

        ir_op = ResizeOp(op_name,
                         transformation_mode=attr_dict["transformation_mode"],
                         interpolation_mode=attr_dict["interpolation_mode"],
                         scale_height=attr_dict["scale_h"],
                         scale_width=attr_dict["scale_w"])
        return ir_op


RelayTranslations.register_translation(RelayUpsamplingTranslation(),
                                       converter_type('upsampling', 'relay'))

