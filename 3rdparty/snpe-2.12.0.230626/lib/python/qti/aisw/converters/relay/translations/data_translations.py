# ==============================================================================
#
#  Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import re
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import (
    CastOp,
    ConcatOp,
    ConstantOp,
    DequantizeOp,
    GatherOp,
    GatherNDOp,
    IdentityOp,
    NeuronOp,
    PackOp,
    QuantizeOp,
    ReshapeOp,
    ResizeOp,
    ScatterNDOp,
    SplitOp,
    StridedSliceOp,
    TileOp,
    TransposeOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase, RelayQuantization

from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay
from tvm.relay.testing import run_infer_type


# ------------------------------------------------------------------------------
#   Cast
# ------------------------------------------------------------------------------
class RelayCastTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCastTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        cast_attrs = relay_expr.attrs
        attr_dict = {}
        attr_dict["to_dtype"] = cast_attrs.dtype
        attr_dict["from_dtype"] = relay_expr.args[0].checked_type.dtype

        log_debug3("\tto_dtype {}", attr_dict["to_dtype"])
        log_debug3("\tfrom_dtype {}", attr_dict["from_dtype"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, CastOp.TRANSLATION_KEY, CastOp.LEGACY_TRANSLATION_KEY)
        to_dtype = attr_dict["to_dtype"]
        from_dtype = attr_dict["from_dtype"]

        ir_op = CastOp(op_name, to_type=to_dtype, from_type=from_dtype)
        return ir_op


RelayTranslations.register_translation(RelayCastTranslation(),
                                       converter_type('cast', 'relay'))


# ------------------------------------------------------------------------------
#   Clip
# ------------------------------------------------------------------------------
class RelayClipTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayClipTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["clip_min"] = relay_expr.attrs.a_min
        attr_dict["clip_max"] = relay_expr.attrs.a_max

        log_debug3("\tclip min {}", attr_dict["clip_min"])
        log_debug3("\tclip max {}", attr_dict["clip_max"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY, NeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = NeuronOp(op_name,
                         neuron_type=ir_graph.QNN_OP_RELU_MIN_MAX,
                         min_value=attr_dict["clip_min"],
                         max_value=attr_dict["clip_max"])
        return ir_op


RelayTranslations.register_translation(RelayClipTranslation(),
                                       converter_type('clip', 'relay'))


# ------------------------------------------------------------------------------
#   Copy
# ------------------------------------------------------------------------------
class RelayCopyTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCopyTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, IdentityOp.TRANSLATION_KEY, IdentityOp.LEGACY_TRANSLATION_KEY)

        return IdentityOp(op_name)


RelayTranslations.register_translation(RelayCopyTranslation(),
                                       converter_type('copy', 'relay'))


# ------------------------------------------------------------------------------
#   Concat
# ------------------------------------------------------------------------------
class RelayConcatTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayConcatTranslation, self).__init__()

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
        op_name = converter_context.get_op_name(relay_expr, ConcatOp.TRANSLATION_KEY, ConcatOp.LEGACY_TRANSLATION_KEY)

        if len(input_names) == 1:
            return IdentityOp(op_name)

        ir_op = ConcatOp(op_name, axis=attr_dict["axis"])
        return ir_op


RelayTranslations.register_translation(RelayConcatTranslation(),
                                       converter_type('concatenate', 'relay'))


# ------------------------------------------------------------------------------
#   Dequantize
# ------------------------------------------------------------------------------
class RelayDequantizeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayDequantizeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        quant_attrs = relay_expr.attrs

        attr_dict = {}
        attr_dict['axis'] = quant_attrs.axis
        m = re.search(r'\d+$', relay_expr.args[0].checked_type.dtype)
        attr_dict['bw'] = int(m.group()) if m else RelayQuantization.DefaultBw
        log_assert(attr_dict['bw'] == 8, "dtype only support uint8 or int8")


        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, DequantizeOp.TRANSLATION_KEY)
        q_params = RelayQuantization.get_quantization_params(op_name, input_names, relay_params, attr_dict)
        q_params['bw'] = attr_dict['bw']
        log_debug3("\tQuantization attributes {}", attr_dict)
        log_debug3("\tQuantization params {}", q_params)

        # Strip the additional quantization inputs
        for name in input_names[1:]:
            input_names.remove(name)

        quir_graph.add_quantization_params(op_name,
                                           output_encodings=q_params)

        return DequantizeOp(op_name, axis=q_params['axis'], bw=q_params['bw'], scale=q_params['scale'], offset=q_params['offset'])


RelayTranslations.register_translation(RelayDequantizeTranslation(),
                                       converter_type('qnn.dequantize', 'relay'))


# ------------------------------------------------------------------------------
#   ExpandDims
# ------------------------------------------------------------------------------
class RelayExpandDimsTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayExpandDimsTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        expand_dims_attrs = relay_expr.attrs
        attr_dict['axis'] = expand_dims_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)
        axis = attr_dict['axis']
        log_debug3("\taxis {}", axis)

        mod = tvm.IRModule.from_expr(relay_expr)
        mod = relay.transform.InferType()(mod)
        output_shape = mod["main"].ret_type.shape
        if isinstance(output_shape, tvm.ir.container.Array):
            log_debug3("\toutput shape {}", output_shape)
            output_shape = [int(x) for x in output_shape]

        ir_op = ReshapeOp(op_name, shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelayExpandDimsTranslation(),
                                       converter_type('expand_dims', 'relay'))


# ------------------------------------------------------------------------------
#   Flatten
# ------------------------------------------------------------------------------
class RelayFlattenTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayFlattenTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        output_shape = list()
        output_shape.append(input_shape[0]) # batch
        output_shape.append(int(np.prod(input_shape[1:])))

        log_debug3("\tOp input shape {}", input_shape)
        log_debug3("\tOp new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelayFlattenTranslation(),
                                       converter_type('batch_flatten', 'relay'))


# ------------------------------------------------------------------------------
#   Full
# ------------------------------------------------------------------------------
class RelayFullTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayFullTranslation, self).__init__()
        self.value = None

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        relay_attrs = relay_expr.attrs
        attr_dict['shape'] = [int(m) for m in relay_attrs.shape]
        attr_dict['value'] = getattr(relay_attrs, 'fill_value', float(self.value))
        attr_dict['dtype'] = relay_attrs.dtype if relay_attrs.dtype else np.float32

        log_debug3("\t shape {}", attr_dict["shape"])
        log_debug3("\t value {}", attr_dict["value"])
        log_debug3("\t dtype {}", attr_dict["dtype"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ConstantOp.TRANSLATION_KEY, ConstantOp.LEGACY_TRANSLATION_KEY)

        shape = attr_dict['shape']
        value = attr_dict['value']
        dtype = attr_dict['dtype']
        tensor = np.full(shape, value, dtype=dtype)

        ir_op = ConstantOp(op_name, tensor=tensor)
        return ir_op


RelayTranslations.register_translation(RelayFullTranslation(),
                                       converter_type('full', 'relay'))


# ------------------------------------------------------------------------------
#   GatherND
# ------------------------------------------------------------------------------
class RelayGatherNDTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayGatherNDTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        attrs = relay_expr.attrs
        attr_dict['batch_dims'] = attrs.batch_dims

        log_debug3("\batch_dims {}", attr_dict["batch_dims"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GatherNDOp.TRANSLATION_KEY, GatherNDOp.LEGACY_TRANSLATION_KEY)

        batch_dims = attr_dict['batch_dims']

        if input_names[0] in relay_params:
            data = relay_params[input_names[0]]
            if isinstance(data, tvm.runtime.ndarray.NDArray) or isinstance(data, tvm.runtime.NDArray):
                data = data.asnumpy()
            log_debug3("\tdata shape {}", data.shape)
            quir_graph.add(ConstantOp(input_names[0], data), [], [input_names[0]])

        # relay frontend will add additional transpose into relay mod since indices of relay.gather_nd is column based,
        # we should add another transpose to negate the transpose added by frontend since QNNIR is row based.
        if input_names[1] in relay_params:
            # if indices is constant, transpose it directly
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            indices_rank = len(indices.shape)
            perm = list(range(1, indices_rank)) + [0]
            indices = np.transpose(indices, perm)
            log_debug3("\tindices shape {}", indices.shape)
            quir_graph.add(ConstantOp(input_names[1], indices), [], [input_names[1]])
        else:
            indices_rank = len(quir_graph.get_buffer(input_names[1]).shape)
            perm = list(range(1, indices_rank)) + [0]
            transpose_op_name = input_names[1] + '_permute'
            transpose_op = TransposeOp(name=transpose_op_name, perm=perm)
            transpose_node = quir_graph.add(transpose_op, input_names=[input_names[1]], output_names=[transpose_op_name])

            input_names[1] = transpose_node.output_names[0]

        ir_op = GatherNDOp(op_name, batch_dims=batch_dims)
        return ir_op


RelayTranslations.register_translation(RelayGatherNDTranslation(),
                                       converter_type('gather_nd', 'relay'))


# ------------------------------------------------------------------------------
#   LayoutTransform
# ------------------------------------------------------------------------------
class RelayLayoutTransformTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayLayoutTransformTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        attr_dict["src_layout"] = relay_expr.attrs.src_layout
        attr_dict["dst_layout"] = relay_expr.attrs.dst_layout

        log_debug3("\t src_layout {}", attr_dict["src_layout"])
        log_debug3("\t dst_layout {}", attr_dict["dst_layout"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TransposeOp.TRANSLATION_KEY,
                                                TransposeOp.LEGACY_TRANSLATION_KEY)

        src_layout = attr_dict["src_layout"]
        dst_layout = attr_dict["dst_layout"]

        permute_order = [src_layout.index(axis_name) for axis_name in dst_layout]

        log_debug3("\t permute_order {}", permute_order)

        return TransposeOp(op_name, permute_order)


RelayTranslations.register_translation(RelayLayoutTransformTranslation(),
                                       converter_type('layout_transform', 'relay'))


# ------------------------------------------------------------------------------
#   Ones
# ------------------------------------------------------------------------------
class RelayOnesTranslation(RelayFullTranslation):
    def __init__(self):
        super(RelayOnesTranslation, self).__init__()
        self.value = 1


RelayTranslations.register_translation(RelayOnesTranslation(),
                                       converter_type('ones', 'relay'))


# ------------------------------------------------------------------------------
#   Quantize
# ------------------------------------------------------------------------------
class RelayQuantizeTranslation(RelayTranslationBase):

    def __init__(self):
        super(RelayQuantizeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        quant_attrs = relay_expr.attrs

        attr_dict = {}
        attr_dict['dtype'] = quant_attrs.out_dtype
        attr_dict['axis'] = quant_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, QuantizeOp.TRANSLATION_KEY)
        q_params = RelayQuantization.get_quantization_params(op_name, input_names, relay_params, attr_dict)

        log_debug3("\tQuantization attributes {}", attr_dict)
        log_debug3("\tQuantization params {}", q_params)

        # Strip the additional quantization inputs
        #input_names = input_names[0:1]
        for name in input_names[1:]:
            input_names.remove(name)

        quir_graph.add_quantization_params(op_name,
                                           output_encodings=q_params)

        return QuantizeOp(op_name, axis=q_params['axis'], bw=q_params['bw'], scale=q_params['scale'], offset=q_params['offset'])


RelayTranslations.register_translation(RelayQuantizeTranslation(),
                                       converter_type('qnn.quantize', 'relay'))


# ------------------------------------------------------------------------------
#   Repeat
# ------------------------------------------------------------------------------
class RelayRepeatTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayRepeatTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["repeats"] = int(relay_expr.attrs.repeats)
        attr_dict["axis"] = int(relay_expr.attrs.axis)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TileOp.TRANSLATION_KEY, TileOp.LEGACY_TRANSLATION_KEY)
        axis = attr_dict["axis"]
        repeats = attr_dict["repeats"]

        input_rank = quir_graph.get_buffer(input_names[0]).rank()

        if axis < 0:
            axis += input_rank

        log_assert(axis < input_rank and axis >= 0, "Axis value shall be less than the number of data dimension")

        multiples = [1]*input_rank
        multiples[axis] = repeats

        ir_op = TileOp(op_name, multiples=multiples)
        return ir_op


RelayTranslations.register_translation(RelayRepeatTranslation(),
                                       converter_type('repeat', 'relay'))


# ------------------------------------------------------------------------------
#   Reshape
# ------------------------------------------------------------------------------
class RelayReshapeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReshapeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["new_shape"] = [int(val) for val in relay_expr.attrs.newshape]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)

        new_shape = attr_dict["new_shape"]
        log_debug3("\tReshape Op attribute new shape {}", new_shape)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        log_debug3("\tReshape Op input shape {}", input_shape)

        ir_op = ReshapeOp(op_name, shape=new_shape)
        return ir_op


RelayTranslations.register_translation(RelayReshapeTranslation(),
                                       converter_type('Reshape', 'relay'))


# ------------------------------------------------------------------------------
#   Resize
# ------------------------------------------------------------------------------
class RelayResizeTranslation(RelayTranslationBase):

    class TransformModes:
        ALIGN_CORNERS = "align_corners"
        ASYMMETRIC = "asymmetric"
        HALF_PIXEL = "half_pixel"

    class ScaleModes:
        BICUBIC = "bicubic"
        BILINEAR = "bilinear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    RELAY_CONSTS_TO_IR = {
        ScaleModes.BILINEAR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR,
        ScaleModes.NEAREST_NEIGHBOR: ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
        TransformModes.ALIGN_CORNERS: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS,
        TransformModes.ASYMMETRIC: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC,
        TransformModes.HALF_PIXEL: ir_graph.QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL,
    }

    def __init__(self):
        super(RelayResizeTranslation, self).__init__()

    def extract_attributes(self, relay_expr: relay.expr.Call, relay_params: dict, **kwargs):
        attr_dict = {}
        resize_attrs = relay_expr.attrs

        attr_dict['size'] = [int(num) for num in getattr(resize_attrs, 'size')]
        attr_dict['layout'] = getattr(resize_attrs, 'layout', 'NCHW')

        log_debug3("\tsize {}", attr_dict['size'])
        log_debug3("\tlayout {}", attr_dict['layout'])

        output_dtype = getattr(resize_attrs, "output_dtype", None)
        if output_dtype is not None:
            raise ValueError("Unsupported conversion to output dtype {} for resize expr".format(output_dtype))

        scale_mode = getattr(resize_attrs, "method", self.ScaleModes.BILINEAR)
        if scale_mode == self.ScaleModes.BICUBIC:
            raise ValueError("Unsupported scale method bicubic for resize expr")
        attr_dict["interpolation_mode"] = self.RELAY_CONSTS_TO_IR[scale_mode]
        log_debug3("\tinterpolation_mode mode {}", attr_dict['interpolation_mode'])

        transform_mode = getattr(resize_attrs, "coordinate_transformation_mode", self.TransformModes.HALF_PIXEL)
        attr_dict["transformation_mode"] = self.RELAY_CONSTS_TO_IR[transform_mode]
        log_debug3("\ttransformation_mode mode {}", attr_dict['transformation_mode'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ResizeOp.TRANSLATION_KEY, ResizeOp.LEGACY_TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]

        if attr_dict['layout'] == 'NHWC':
            output_shape = [input_shape[0], attr_dict['size'][0], attr_dict['size'][1], input_shape[3]]
        else:
            raise ValueError("Unknown data layout {}".format(attr_dict['layout']))

        scale_height = output_shape[-3] / input_shape[-3]
        scale_width = output_shape[-2] / input_shape[-2]

        ir_op = ResizeOp(op_name,
                         transformation_mode=attr_dict["transformation_mode"],
                         interpolation_mode=attr_dict["interpolation_mode"],
                         scale_height=scale_height,
                         scale_width=scale_width)
        return ir_op


RelayTranslations.register_translation(RelayResizeTranslation(),
                                       converter_type('resize', 'relay'))


# ------------------------------------------------------------------------------
#   Resize2D
# ------------------------------------------------------------------------------
class RelayResize2DTranslation(RelayResizeTranslation):

    class TransformModes:
        ALIGN_CORNERS = "align_corners"
        ASYMMETRIC = "asymmetric"
        HALF_PIXEL = "half_pixel"

    class ScaleModes:
        BICUBIC = "cubic"
        BILINEAR = "linear"
        NEAREST_NEIGHBOR = "nearest_neighbor"

    RELAY_CONSTS_TO_IR = {
        ScaleModes.BILINEAR: "bilinear",
        ScaleModes.NEAREST_NEIGHBOR: "nearest"
    }

    def __init__(self):
        super(RelayResize2DTranslation, self).__init__()

    def extract_attributes(self, relay_expr: relay.expr.Call, relay_params: dict, **kwargs):
        resize_attrs = relay_expr.attrs
        attr_dict = super().extract_attributes(relay_expr, relay_params)

        rounding_method = getattr(resize_attrs, "rounding_method")
        log_debug3("\trounding method {}", rounding_method)

        return attr_dict


RelayTranslations.register_translation(RelayResize2DTranslation(),
                                       converter_type('resize2d', 'relay'))


# ------------------------------------------------------------------------------
#   Reverse
# ------------------------------------------------------------------------------
class RelayReverseTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayReverseTranslation, self).__init__()

    def extract_attributes(self, relay_expr: relay.expr.Call, relay_params: dict, **kwargs):

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

        input_shape = quir_graph.get_buffer(input_names[0]).shape
        reverse_axis = attr_dict["axis"]

        begin = [0]*len(input_shape)
        end = list(input_shape)
        strides = [1]*len(input_shape)

        # We use StridedSlice with stride = -1, end = -1 to mimic the effect of reverse.
        # For the axis which is reversed, (begin, end, stride) = (dim[axis]-1, -1, -1).
        # Otherwise, (begin, end, stride) = (0, dim[axis], 1)
        begin[reverse_axis] = input_shape[reverse_axis] - 1
        end[reverse_axis] = -1
        strides[reverse_axis] = -1

        op_name = converter_context.get_op_name(relay_expr, StridedSliceOp.TRANSLATION_KEY,
                                                StridedSliceOp.LEGACY_TRANSLATION_KEY)
        ranges = list(map(list, zip(begin, end, strides)))
        ir_op = StridedSliceOp(op_name, ranges=ranges)
        return ir_op


RelayTranslations.register_translation(RelayReverseTranslation(),
                                       converter_type('reverse', 'relay'))


# ------------------------------------------------------------------------------
#   ScatterND
# ------------------------------------------------------------------------------
class RelayScatterNDTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayScatterNDTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        scatternd_attrs = relay_expr.attrs
        attr_dict['mode'] = "none"
        if scatternd_attrs.mode == 'add':
            attr_dict['mode'] = "add"
        elif scatternd_attrs.mode != 'update':
            raise TypeError("Unsupported mode for scatter_nd: {}".format(scatternd_attrs.mode))
        log_debug3("\taccumulation mode {}", attr_dict['mode'])
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ScatterNDOp.TRANSLATION_KEY)

        if input_names[1] in relay_params:
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            log_debug3("\tindices shape {}", indices.shape)
            indices_output = input_names[1]
            # we don't need to populate quantization params for input[1] since it is indices and its dtype is int32
            quir_graph.add(ConstantOp(indices_output, indices), [], [indices_output])
        if input_names[2] in relay_params:
            updates = relay_params[input_names[2]]
            if isinstance(updates, tvm.runtime.ndarray.NDArray) or isinstance(updates, tvm.runtime.NDArray):
                updates = updates.asnumpy()
            log_debug3("\tupdates shape {}", updates.shape)
            updates_output = input_names[2]
            self.populate_quantization_params(relay_expr.args[2], converter_context, quir_graph, [updates_output], is_param=True)
            quir_graph.add(ConstantOp(updates_output, updates), [], [updates_output])

        ir_op = ScatterNDOp(op_name, reduction=attr_dict['mode'])
        return ir_op

RelayTranslations.register_translation(RelayScatterNDTranslation(),
                                       converter_type('scatter_nd', 'relay'))


# ------------------------------------------------------------------------------
#   Split
# ------------------------------------------------------------------------------
class RelaySplitTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySplitTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict["axis"] = int(relay_expr.attrs.axis)
        attr_dict["slice_points"] = relay_expr.attrs.indices_or_sections

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, SplitOp.TRANSLATION_KEY, SplitOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict["axis"]
        slice_points = attr_dict["slice_points"]

        output_shapes = []
        slices = []
        num_outputs = 0

        input_shapes = converter_context.get_input_shapes(relay_expr)
        slice_input_shape = input_shapes[0][:]
        if isinstance(slice_points, tvm.ir.container.Array):
            log_debug3("\tslice points {}", slice_points)
            num_outputs = len(slice_points) + 1
            slices = [int(val) for val in slice_points]

            log_debug3("\tmax dim {}", slice_input_shape[axis])
            slice_sizes = [0] + slices + [slice_input_shape[axis]]
            log_debug3("\tslice sizes {}", slice_sizes)

            for i in range(num_outputs):
                output_shapes.append(slice_input_shape[:])
                output_shapes[i][axis] = slice_sizes[i + 1] - slice_sizes[i]
        elif isinstance(slice_points, tvm.tir.expr.IntImm):
            log_debug3("\tslice points {}", int(slice_points))
            num_outputs = int(slice_points)

            # IR can handle [] and split the output evenly using the num of outputs
            slices = []

            for i in range(num_outputs):
                output_shapes.append(input_shapes[0][:])
                output_shapes[i][axis] = int(int(output_shapes[i][axis]) / num_outputs)
        else:
            raise TypeError("Unsupported type {} for slice_points in SplitOp".format(type(slice_points)))

        log_debug3("\tnum_outputs {}", num_outputs)
        log_debug3("\tslices {}", slices)
        log_debug3("\toutput shapes {}", output_shapes)

        ir_op = SplitOp(op_name, axis=axis, split_index=slices, output_shape=output_shapes)
        ir_op.num_outputs = num_outputs
        return ir_op


RelayTranslations.register_translation(RelaySplitTranslation(),
                                       converter_type('Split', 'relay'))


# ------------------------------------------------------------------------------
#   Squeeze
# ------------------------------------------------------------------------------
class RelaySqueezeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelaySqueezeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        squeeze_attrs = relay_expr.attrs
        attr_dict["axis"] = squeeze_attrs.axis

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ReshapeOp.TRANSLATION_KEY, ReshapeOp.LEGACY_TRANSLATION_KEY)

        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        log_debug3("\tSqueeze Op input shape {}", input_shape)

        axis = attr_dict['axis']

        if axis is None:
            output_shape = [dim for dim in input_shape if dim != 1]
        else:
            axis = [ax % len(input_shape) for ax in axis]
            log_debug3("\taxis {}", axis)

            output_shape = []
            for index, shape in enumerate(input_shape):
                if index in axis:
                    if shape != 1:
                        raise ValueError("Input shape {} at axis {} should be 1", input_shape, index)
                    continue
                output_shape.append(shape)
        log_debug3("\tSqueeze Op new shape {}", output_shape)

        ir_op = ReshapeOp(op_name, shape=output_shape)
        return ir_op


RelayTranslations.register_translation(RelaySqueezeTranslation(),
                                       converter_type('squeeze', 'relay'))


# ------------------------------------------------------------------------------
#   Stack
# ------------------------------------------------------------------------------
class RelayStackTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayStackTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        stack_attrs = relay_expr.attrs
        attr_dict["axis"] = stack_attrs.axis
        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, PackOp.TRANSLATION_KEY, PackOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict["axis"]
        if isinstance(axis, tvm.tir.expr.IntImm):
            axis = int(axis)

        ir_op = PackOp(op_name, axis=axis)

        return ir_op


RelayTranslations.register_translation(RelayStackTranslation(),
                                       converter_type('stack', 'relay'))


# ------------------------------------------------------------------------------
#   StridedSlice
# ------------------------------------------------------------------------------
class RelayStridedSliceTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayStridedSliceTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        strided_slice_attrs = relay_expr.attrs
        attr_dict['begin'] = strided_slice_attrs.begin
        attr_dict['end'] = strided_slice_attrs.end
        attr_dict['strides'] = strided_slice_attrs.strides
        attr_dict['slice_mode'] = strided_slice_attrs.slice_mode
        attr_dict['axes'] = strided_slice_attrs.axes

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, StridedSliceOp.TRANSLATION_KEY,
                                                StridedSliceOp.LEGACY_TRANSLATION_KEY)

        begin = attr_dict['begin']
        end = attr_dict['end']
        strides = attr_dict['strides']
        slice_mode = attr_dict['slice_mode']
        axes = attr_dict['axes']
        input_shape = converter_context.get_input_shapes(relay_expr)[0]

        # axes param is added in tvm v0.8
        # this check will be removed once the axes supported is added
        if axes is not None:
            raise ValueError("Unsupported axes value {} in StridedSliceOp".format(axes))

        if slice_mode == 'size':
            raise ValueError("Unsupported slice mode {} in StridedSliceOp".format(slice_mode))

        if isinstance(begin, tvm.ir.container.Array):
            begin = [int(begin_points) for begin_points in begin]
            input_dim = quir_graph.get_buffer(input_names[0]).shape
            begin = [begin_points + int(input_dim[i]) if begin_points < 0 else begin_points
                     for i, begin_points
                     in enumerate(begin)]
        elif isinstance(begin, tvm.tir.expr.IntImm):
            begin = int(begin)
        else:
            raise TypeError("Unsupported type {} for begin in StridedSliceOp".format(type(begin)))

        if isinstance(strides, tvm.ir.container.Array):
            strides = [int(strides_points) for strides_points in strides]
        elif isinstance(strides, tvm.tir.expr.IntImm):
            strides = int(strides)
        else:
            raise TypeError("Unsupported type {} for strides in StridedSliceOp".format(type(strides)))

        if isinstance(end, tvm.ir.container.Array):
            end = [int(end_points) for end_points in end]
            input_dim = quir_graph.get_buffer(input_names[0]).shape
            # -1 is valid for endpoint if the stride is negative and does not need to be changed
            end = [end_points + int(input_dim[i])
                       if (end_points < 0 and strides[i] > 0) or (end_points < -1 and strides[i] < 0)
                       else end_points
                   for i, end_points
                   in enumerate(end)]
        elif isinstance(end, tvm.tir.expr.IntImm):
            end = int(end)
        else:
            raise TypeError("Unsupported type {} for end in StridedSliceOp".format(type(end)))


        log_debug3("\tbegin {}", begin)
        log_debug3("\tend {}", end)
        log_debug3("\tstrides {}", strides)
        log_debug3("\tslice_mode {}", slice_mode)

        if len(strides) == 1 and len(strides) < len(begin):
            strides = strides * len(begin)

        if len(begin) < len(input_shape):
            raise ValueError("Unsupported length for begin in StridedSliceOp. Expected {}. Got {}."
                             .format(len(input_shape), len(begin)))
        if len(end) < len(input_shape):
            raise ValueError("Unsupported length for end in StridedSliceOp. Expected {}. Got {}."
                             .format(len(input_shape), len(end)))
        if len(strides) < len(input_shape):
            raise ValueError("Unsupported length for strides in StridedSliceOp. Expected {}. Got {}."
                             .format(len(input_shape), len(strides)))

        ranges = list(map(list, zip(begin, end, strides)))
        ir_op = StridedSliceOp(op_name, ranges=ranges)
        return ir_op


RelayTranslations.register_translation(RelayStridedSliceTranslation(),
                                       converter_type('strided_slice', 'relay'))


# ------------------------------------------------------------------------------
#   Take
# ------------------------------------------------------------------------------
class RelayTakeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTakeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        gather_attrs = relay_expr.attrs
        attr_dict['axis'] = gather_attrs.axis

        log_debug3("\taxis {}", attr_dict["axis"])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, GatherOp.TRANSLATION_KEY, GatherOp.LEGACY_TRANSLATION_KEY)

        axis = attr_dict['axis']
        if input_names[1] in relay_params:
            indices = relay_params[input_names[1]]
            if isinstance(indices, tvm.runtime.ndarray.NDArray) or isinstance(indices, tvm.runtime.NDArray):
                indices = indices.asnumpy()
            log_debug3("\tindices {}", indices)
            indices_output = input_names[1]
            # we don't need to populate quantization params for input[1] since it is indices and its dtype is int32
            quir_graph.add(ConstantOp(indices_output, indices), [], [indices_output])

        if input_names[0] in relay_params:
            data = relay_params[input_names[0]]
            if isinstance(data, tvm.runtime.ndarray.NDArray) or isinstance(data, tvm.runtime.NDArray):
                data = data.asnumpy()
            log_debug3("\tdata {}", data)
            data_output = input_names[0]
            self.populate_quantization_params(relay_expr.args[0], converter_context, quir_graph, [data_output], is_param=True)
            quir_graph.add(ConstantOp(data_output, data), [], [data_output])

        ir_op = GatherOp(op_name, axis=axis)
        return ir_op


RelayTranslations.register_translation(RelayTakeTranslation(),
                                       converter_type('take', 'relay'))


# ------------------------------------------------------------------------------
#   Tile
# ------------------------------------------------------------------------------
class RelayTileTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTileTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}

        tile_attrs = relay_expr.attrs
        attr_dict['multiples'] = [int(m) for m in tile_attrs.reps]

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TileOp.TRANSLATION_KEY, TileOp.LEGACY_TRANSLATION_KEY)

        multiples =  attr_dict['multiples']

        log_assert(all([m > 0 for m in multiples]), "Multiples {} shall be all postive value", multiples)

        ir_op = TileOp(op_name, multiples=multiples)
        return ir_op


RelayTranslations.register_translation(RelayTileTranslation(),
                                       converter_type('tile', 'relay'))


# ------------------------------------------------------------------------------
#   Transpose
# ------------------------------------------------------------------------------
class RelayTransposeTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTransposeTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        transpose_attr = relay_expr.attrs
        axes = transpose_attr.axes if hasattr(transpose_attr, 'axes') else None
        attr_dict['axes'] = axes

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TransposeOp.TRANSLATION_KEY,
                                                TransposeOp.LEGACY_TRANSLATION_KEY)

        if attr_dict['axes'] is None:
            # reverse order if not specified
            input_shape = converter_context.get_input_shapes(relay_expr)[0]
            input_dimensions = len(input_shape)
            axes = [i for i in reversed(range(input_dimensions))]
        else:
            axes = [int(i) for i in attr_dict['axes']]

        log_debug3("\taxes {}", axes)

        return TransposeOp(op_name, axes)


RelayTranslations.register_translation(RelayTransposeTranslation(),
                                       converter_type('transpose', 'relay'))


# ------------------------------------------------------------------------------
#   Zeros
# ------------------------------------------------------------------------------
class RelayZerosTranslation(RelayFullTranslation):
    def __init__(self):
        super(RelayZerosTranslation, self).__init__()
        self.value = 0


RelayTranslations.register_translation(RelayZerosTranslation(),
                                       converter_type('zeros', 'relay'))
