# ==============================================================================
#
#  Copyright (c) 2018-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from .onnx_translations import *

import distutils
from distutils import version
import packaging

from qti.aisw.converters.common import ir_graph


# ------------------------------------------------------------------------------
#   Abs
# ------------------------------------------------------------------------------
class OnnxAbsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Abs', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ABS)


OnnxTranslations.register_translation(OnnxAbsTranslation(),
                                      converter_type('Abs', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ABS])


# ------------------------------------------------------------------------------
#   Add
# ------------------------------------------------------------------------------
class OnnxAddTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Add', [1, 6, 7])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_ADD
        self.numpy_op = numpy.add


OnnxTranslations.register_translation(OnnxAddTranslation(),
                                      converter_type('Add', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD])


# ------------------------------------------------------------------------------
#   And
# ------------------------------------------------------------------------------
class OnnxAndTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('And', [1, 7])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_AND
        self.numpy_op = numpy.logical_and


OnnxTranslations.register_translation(OnnxAndTranslation(),
                                      converter_type('And', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND])


# ------------------------------------------------------------------------------
#   ArgMax, ArgMin
# ------------------------------------------------------------------------------
class OnnxArgOpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ArgMax', [1, 11])
        self.register_op_schema('ArgMin', [1, 11])

    def extract_parameters(self, src_op, converter_context):
        # these parameters depends on src_op.op_type(ArgMax/ArgMin)
        params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type), validate=True)

        if str(src_op.op_type) == 'ArgMax':
            arg_type = ir_graph.QNN_OP_ARGMAX
        elif str(src_op.op_type) == 'ArgMin':
            arg_type = ir_graph.QNN_OP_ARGMIN

        return op_adapter.ArgOp(str(src_op.name),
                                arg_type = arg_type,
                                axis=params.axis,
                                keep_dims=params.keepdims)


OnnxTranslations.register_translation(OnnxArgOpTranslation(),
                                      converter_type('ArgMax', 'onnx'),
                                      converter_type('ArgMin', 'onnx'),
                                      op_adapter.ArgOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Asin
# ------------------------------------------------------------------------------
class OnnxAsinTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Asin', [7])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ASIN)


OnnxTranslations.register_translation(OnnxAsinTranslation(),
                                      converter_type('Asin', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ASIN])


# ------------------------------------------------------------------------------
#   Atan
# ------------------------------------------------------------------------------
class OnnxAtanTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Atan', [7])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ATAN)


OnnxTranslations.register_translation(OnnxAtanTranslation(),
                                      converter_type('Atan', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ATAN])


# ------------------------------------------------------------------------------
#   Ceil
# ------------------------------------------------------------------------------
class OnnxCeilTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Ceil', [1, 6, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_CEIL)


OnnxTranslations.register_translation(OnnxCeilTranslation(),
                                      converter_type('Ceil', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_CEIL])


# ------------------------------------------------------------------------------
#   Cos
# ------------------------------------------------------------------------------
class OnnxCosTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Cos', [7])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_COS)


OnnxTranslations.register_translation(OnnxCosTranslation(),
                                      converter_type('Cos', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_COS])


# ------------------------------------------------------------------------------
#   CumSum
# ------------------------------------------------------------------------------
class OnnxCumSumTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('CumSum', [11, 14])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()

        # extract axis param
        const_op = self.fetch_constant_op(input_names[1], converter_context, fail_if_dynamic=True)
        axis = const_op.tensor.astype(numpy.int32).item(0)

        # check for axis to be in [-r,r-1]
        if axis not in range(-input_rank, input_rank):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(axis, input_names[1], src_op.op_type))
        if axis < 0:
            axis += input_rank

        # extract reverse and exclusive params
        reverse = params.reverse if 'reverse' in params else 0
        if reverse not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(reverse, "reverse", src_op.op_type))
        exclusive = params.exclusive if 'exclusive' in params else 0
        if exclusive not in (0, 1):
            raise ValueError("ERROR: Invalid value {} for {} attribute for {} op".format(exclusive, "exclusive", src_op.op_type))

        # axis received as input, but added as param in our IR graph
        return op_adapter.CumSumOp(str(src_op.name),
                                   axis=axis,
                                   reverse=bool(reverse),
                                   exclusive=bool(exclusive))

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]

OnnxTranslations.register_translation(OnnxCumSumTranslation(),
                                      converter_type('CumSum', 'onnx'),
                                      op_adapter.CumSumOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Div
# ------------------------------------------------------------------------------
class OnnxDivTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Div', [1, 6, 7])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE
        self.numpy_op = numpy.divide


OnnxTranslations.register_translation(OnnxDivTranslation(),
                                      converter_type('Div', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE])


# ------------------------------------------------------------------------------
#   Elu
# ------------------------------------------------------------------------------
class OnnxEluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Elu', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        # these parameters belong to Elu
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_neuron_type(src_op.op_type),
                                   alpha=params.alpha)


OnnxTranslations.register_translation(OnnxEluTranslation(),
                                      converter_type('Elu', 'onnx'))


# ------------------------------------------------------------------------------
#   Equal
# ------------------------------------------------------------------------------
class OnnxEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Equal', [1, 7, 11])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_EQUAL
        self.numpy_op = numpy.equal


OnnxTranslations.register_translation(OnnxEqualTranslation(),
                                      converter_type('Equal', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EQUAL])


# ------------------------------------------------------------------------------
#   Erf
# ------------------------------------------------------------------------------
class OnnxErfTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Erf', [9, 13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ErfOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxErfTranslation(),
                                      converter_type('Erf', 'onnx'),
                                      op_adapter.ErfOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Exp
# ------------------------------------------------------------------------------
class OnnxExpTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Exp', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_EXP)


OnnxTranslations.register_translation(OnnxExpTranslation(),
                                      converter_type('Exp', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EXP])


# ------------------------------------------------------------------------------
#   Floor
# ------------------------------------------------------------------------------
class OnnxFloorTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Floor', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_FLOOR)


OnnxTranslations.register_translation(OnnxFloorTranslation(),
                                      converter_type('Floor', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FLOOR])


# ------------------------------------------------------------------------------
#   GEMM
# ------------------------------------------------------------------------------
class OnnxGemmTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Gemm', [1, 6, 7, 9, 11])
        self.params = None

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        log_warning(code_to_message.get_warning_message("WARNING_GEMM"))
        self.params = extract_attributes(src_op, schema=self.op_schema(), validate=True)
        input_names = list(map(str, src_op.input))
        bias = None
        # In newer opset versions, bias is made an optional parameter
        # in the Gemm operator. Default value of bias in this case is 0
        weights = converter_context.weights.fetch(input_names[1])
        if len(src_op.input) == 3:
            bias = converter_context.weights.fetch(input_names[2])
        weights = weights * self.params.alpha

        # Transpose weights if transB is given
        if self.params.transB:
            weights = numpy.ascontiguousarray(numpy.transpose(weights, (1, 0)))

        if bias is None:
            bias = numpy.zeros((weights.shape[1],))

        bias = bias * self.params.beta

        # Transpose input if transA is given
        if self.params.transA:
            permute_op = op_adapter.TransposeOp(input_names[0] + '_permute', perm=[1, 0])
            graph.add(permute_op, [input_names[0]], [input_names[0] + '_permute'])
            graph.add_src_op_info(permute_op.name, [input_names[0]], [input_names[0] + '_permute'])
        self.weights_name = input_names[1]

        if not graph.has_buffer(self.weights_name):
            weights_constant_op = op_adapter.ConstantOp(self.weights_name, tensor=weights)
            weight_node = graph.add(weights_constant_op, [], [self.weights_name])
            graph.add_src_op_info(self.weights_name, None, weight_node.output_names[0])
        if len(src_op.input) == 3:
            self.bias_name = input_names[2]
        else:
            self.bias_name = input_names[0] + '_b'
        if not graph.has_buffer(self.bias_name):
            bias_constant_op = op_adapter.ConstantOp(self.bias_name, tensor=bias)
            bias_node = graph.add(bias_constant_op, [], [self.bias_name])
            graph.add_src_op_info(self.bias_name, None, bias_node.output_names[0])

        return op_adapter.FullyConnectedOp(str(src_op.name),
                                           bias_op_name=self.bias_name)

    def extract_input_names(self, src_op, converter_context):
        if self.params.transA:
            return [str(src_op.input[0]) + '_permute', self.weights_name, self.bias_name]
        return [src_op.input[0], self.weights_name, self.bias_name]


OnnxTranslations.register_translation(OnnxGemmTranslation(), converter_type('Gemm', 'onnx'))


# ------------------------------------------------------------------------------
#   Greater
# ------------------------------------------------------------------------------
class OnnxGreaterTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Greater', [1, 7, 9])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_GREATER
        self.numpy_op = numpy.greater


OnnxTranslations.register_translation(OnnxGreaterTranslation(),
                                      converter_type('Greater', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER])


# ------------------------------------------------------------------------------
#   GreaterOrEqual
# ------------------------------------------------------------------------------
class OnnxGreaterOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('GreaterOrEqual', [12])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL
        self.numpy_op = numpy.greater_equal


# GreaterOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) >= packaging.version.Version("1.7.0"):
    OnnxTranslations.register_translation(OnnxGreaterOrEqualTranslation(),
                                          converter_type('GreaterOrEqual', 'onnx'),
                                          op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL])


# ------------------------------------------------------------------------------
#   Identity
# ------------------------------------------------------------------------------
class OnnxIdentityTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Identity', [1])

    def extract_parameters(self, src_op, converter_context):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op. Otherwise the identity op is a no-op that
        # gets squashed later.
        graph = converter_context.ir_graph
        if not graph.has_buffer(src_op.input[0]):
            const_input = converter_context.weights.fetch(str(src_op.input[0]))
            converter_context.weights.insert(str(src_op.output[0]), const_input)
            return op_adapter.ConstantOp(src_op.output[0], const_input)

        return op_adapter.IdentityOp(str(src_op.name))

    def extract_input_names(self, src_op, converter_context):
        # if the input buffer is not in the graph, that means
        # it is a const input. We replace all const inputs with a
        # const op which do not need an input name.
        if not converter_context.ir_graph.has_buffer(src_op.input[0]):
            return []
        return str(src_op.input[0])


OnnxTranslations.register_translation(OnnxIdentityTranslation(),
                                      converter_type('Identity', 'onnx'))


# ------------------------------------------------------------------------------
#   Less
# ------------------------------------------------------------------------------
class OnnxLessTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Less', [1, 7, 9])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_LESS
        self.numpy_op = numpy.less


OnnxTranslations.register_translation(OnnxLessTranslation(),
                                      converter_type('Less', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS])


# ------------------------------------------------------------------------------
#   LessOrEqual
# ------------------------------------------------------------------------------
class OnnxLessOrEqualTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LessOrEqual', [12])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL
        self.numpy_op = numpy.less_equal


# LessOrEqual is announced in ONNX 1.7.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) >= packaging.version.Version("1.7.0"):
    OnnxTranslations.register_translation(OnnxLessOrEqualTranslation(),
                                          converter_type('LessOrEqual', 'onnx'),
                                          op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL])


# ------------------------------------------------------------------------------
#   LpNormalization
# ------------------------------------------------------------------------------
class OnnxLpNormalizationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LpNormalization', [1])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())

        if params.p != 2:
            raise ValueError("Only the L2-Norm is supported. "
                             "Found order of {}".format(params.p))

        # we use the default value of epsilon here
        return op_adapter.L2NormOp(src_op.name,
                                   axis=params.axis)


OnnxTranslations.register_translation(OnnxLpNormalizationTranslation(),
                                      converter_type('LpNormalization', 'onnx'),
                                      op_adapter.L2NormOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Log
# ------------------------------------------------------------------------------
class OnnxLogTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Log', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_LOG)


OnnxTranslations.register_translation(OnnxLogTranslation(),
                                      converter_type('Log', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LOG])


# ------------------------------------------------------------------------------
#   LogSoftmax
# ------------------------------------------------------------------------------
class OnnxLogSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('LogSoftmax', [1, 11])

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        axis = getattr(params, "axis", 1)
        return op_adapter.LogSoftmaxOp(str(src_op.name),
                                       axis=axis)


OnnxTranslations.register_translation(OnnxLogSoftmaxTranslation(),
                                      converter_type('LogSoftmax', 'onnx'),
                                      op_adapter.LogSoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Matmul
# ------------------------------------------------------------------------------
class OnnxMatMulTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('MatMul', [1, 9, 13])
        self.input_names = []
        self.output_names = []

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        op = self.extract_parameters(src_op, converter_context)
        if not op:
            # Return last node
            return graph.list_nodes()[-1]

        node = graph.add(op, self.input_names, self.output_names)
        self.add_src_op_info(node.op.name, src_op, graph)

        return node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.input_names = self.extract_input_names(src_op, converter_context)
        self.output_names = self.extract_output_names(src_op, converter_context)

        # If input A is static, add a Constant Op
        if converter_context.weights.has(self.input_names[0]):
            static_input_name = self.input_names[0]
            tensor = converter_context.weights.fetch(static_input_name, prunable=False)
            if not graph.has_buffer(static_input_name):
                static_input_op = op_adapter.ConstantOp(static_input_name, tensor=tensor)
                static_input_node = graph.add(static_input_op, [], [static_input_name])
                graph.add_src_op_info(static_input_node.op.name, [], [static_input_name])

        # If input B is constant, add a Constant Op
        if converter_context.weights.has(self.input_names[1]):
            weight_input_name = self.input_names[1]
            weights = converter_context.weights.fetch(weight_input_name, prunable=False)
            if not graph.has_buffer(weight_input_name):
                weights_constant_op = op_adapter.ConstantOp(weight_input_name, tensor=weights)
                weight_node = graph.add(weights_constant_op, [], [weight_input_name])
                graph.add_src_op_info(weight_node.op.name, [], [weight_input_name])

        # Since ONNX Matmul does not support matrix transpose,
        # both transpose_in0 and transpose_in1 are set False
        return op_adapter.MatMulOp(name=str(src_op.name),
                                   transpose_in0=False,
                                   transpose_in1=False)


OnnxTranslations.register_translation(OnnxMatMulTranslation(), converter_type('MatMul', 'onnx'))


# ------------------------------------------------------------------------------
#   Max
# ------------------------------------------------------------------------------
class OnnxMaxTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Max', [1, 6, 8, 12])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM
        self.numpy_op = numpy.maximum


OnnxTranslations.register_translation(OnnxMaxTranslation(),
                                      converter_type('Max', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM])


# ------------------------------------------------------------------------------
#   Min
# ------------------------------------------------------------------------------
class OnnxMinTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Min', [1, 6, 8])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM
        self.numpy_op = numpy.minimum


OnnxTranslations.register_translation(OnnxMinTranslation(),
                                      converter_type('Min', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM])


# ------------------------------------------------------------------------------
#   Mod
# ------------------------------------------------------------------------------
class OnnxModTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mod', [10, 13])
        self.numpy_op = numpy.mod
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_MOD

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op, schema=self.op_schema())
        fmod = getattr(params, "fmod", 0)
        if fmod:
            self.numpy_op = numpy.fmod
            self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_FMOD

        # reuse super function to perform constant folding and return op
        return super().extract_parameters(src_op, converter_context)


OnnxTranslations.register_translation(OnnxModTranslation(),
                                      converter_type('Mod', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MOD],
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FMOD])


# ------------------------------------------------------------------------------
#   Mul
# ------------------------------------------------------------------------------
class OnnxMulTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Mul', [1, 6, 7])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY
        self.numpy_op = numpy.multiply


OnnxTranslations.register_translation(OnnxMulTranslation(),
                                      converter_type('Mul', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY])


# ------------------------------------------------------------------------------
#   Neg
# ------------------------------------------------------------------------------
class OnnxNegTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Neg', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        input_names = list(map(str, src_op.input))
        const_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False,
                                          fail_if_dynamic=False, fail_if_not_found=True)
        if const_op:
            log_debug1("Node {} with static input(s) is resolved as Constant Op and interpreted during conversion".format(str(src_op.name)))
            data = const_op.tensor
            neg_data = np.negative(data)
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in input_names])
            converter_context.weights.insert(str(src_op.output[0]), neg_data, was_scalar=was_scalar)
            return None

        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_NEG)


OnnxTranslations.register_translation(OnnxNegTranslation(),
                                      converter_type('Neg', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NEG])


# ------------------------------------------------------------------------------
#   Not
# ------------------------------------------------------------------------------
class OnnxNotTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Not', [1])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_NOT)


OnnxTranslations.register_translation(OnnxNotTranslation(),
                                      converter_type('Not', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NOT])


# ------------------------------------------------------------------------------
#   Or
# ------------------------------------------------------------------------------
class OnnxOrTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Or', [1, 7])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_OR
        self.numpy_op = numpy.logical_or


OnnxTranslations.register_translation(OnnxOrTranslation(),
                                      converter_type('Or', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_OR])


# ------------------------------------------------------------------------------
#   Pow
# ------------------------------------------------------------------------------
class OnnxPowTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Pow', [1, 7, 12])

    def extract_parameters(self, src_op, converter_context):

        graph = converter_context.ir_graph
        power_input_name = src_op.input[1]
        power_op = self.fetch_constant_op(power_input_name, converter_context, prunable=False, fail_if_dynamic=False)
        if power_op and not graph.has_buffer(power_input_name):
            graph.add(power_op, [], power_input_name)

        return op_adapter.ElementwiseBinaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_POWER)


OnnxTranslations.register_translation(OnnxPowTranslation(),
                                      converter_type('Pow', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_POWER])


# ------------------------------------------------------------------------------
#   ReduceBase
# ------------------------------------------------------------------------------
class OnnxReduceBaseTranslation(OnnxTranslationBase):
    def __init__(self, reduce_type):
        OnnxTranslationBase.__init__(self)
        self.reduce_type = reduce_type

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        input_name = src_op.input[0]
        const_op = self.fetch_constant_op(input_name, converter_context, fail_if_dynamic=False)
        if const_op and not graph.has_buffer(input_name):
            graph.add(const_op, [], input_name)

        input_buf = graph.get_buffer(input_name)
        schema = self.op_schema()
        schema.replace_default_values(axes=range(input_buf.rank()))
        params = extract_attributes(src_op, schema=schema)

        return op_adapter.ReduceOp(str(src_op.name),
                                   reduce_type=self.reduce_type,
                                   axes=params.axes,
                                   keep_dims=params.keepdims)

    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]


# ------------------------------------------------------------------------------
#   ReduceL2
# ------------------------------------------------------------------------------
class OnnxReduceL2Translation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.IR_OP_REDUCE_L2)
        self.register_op_schema('ReduceL2', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceL2Translation(),
                                      converter_type('ReduceL2', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.IR_OP_REDUCE_L2])


# ------------------------------------------------------------------------------
#   ReduceMax
# ------------------------------------------------------------------------------
class OnnxReduceMaxTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MAX)
        self.register_op_schema('ReduceMax', [1, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMaxTranslation(),
                                      converter_type('ReduceMax', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX])


# ------------------------------------------------------------------------------
#   ReduceMean
# ------------------------------------------------------------------------------
class OnnxReduceMeanTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MEAN)
        self.register_op_schema('ReduceMean', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMeanTranslation(),
                                      converter_type('ReduceMean', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN])


# ------------------------------------------------------------------------------
#   ReduceMin
# ------------------------------------------------------------------------------
class OnnxReduceMinTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_MIN)
        self.register_op_schema('ReduceMin', [1, 11, 12, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceMinTranslation(),
                                      converter_type('ReduceMin', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN])


# ------------------------------------------------------------------------------
#   ReduceProd
# ------------------------------------------------------------------------------
class OnnxReduceProdTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_PROD)
        self.register_op_schema('ReduceProd', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceProdTranslation(),
                                      converter_type('ReduceProd', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD])


# ------------------------------------------------------------------------------
#   ReduceSum
# ------------------------------------------------------------------------------
class OnnxReduceSumTranslation(OnnxReduceBaseTranslation):
    def __init__(self):
        OnnxReduceBaseTranslation.__init__(self, ir_graph.QNN_OP_REDUCE_SUM)
        self.register_op_schema('ReduceSum', [1, 11, 13])

    def extract_parameters(self, src_op, converter_context):
        return OnnxReduceBaseTranslation.extract_parameters(self, src_op, converter_context)


OnnxTranslations.register_translation(OnnxReduceSumTranslation(),
                                      converter_type('ReduceSum', 'onnx'),
                                      op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM])


# ------------------------------------------------------------------------------
#   Relu
# ------------------------------------------------------------------------------
class OnnxReluTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Relu', [1, 6])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        constant_input_op, op = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        if constant_input_op is not None and not graph.has_buffer(input_names[0]):
            graph.add(constant_input_op, [], [input_names[0]])
            graph.add_src_op_info(constant_input_op.name, [], [input_names[0]])

        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node

    def extract_parameters(self, src_op, converter_context):
        input_names = self.extract_input_names(src_op, converter_context)
        # in case the input to the relu is of type initializer
        constant_input_op = self.fetch_constant_op(input_names[0], converter_context, prunable=False, fail_if_dynamic=False)

        op = op_adapter.NeuronOp(str(src_op.name),
                                 op_adapter.NeuronOp.extract_neuron_type(src_op.op_type))
        return constant_input_op, op


OnnxTranslations.register_translation(OnnxReluTranslation(),
                                      converter_type('Relu', 'onnx'),
                                      op_adapter.NeuronOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Round
# ------------------------------------------------------------------------------
class OnnxRoundTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Round', [11])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ROUND)


OnnxTranslations.register_translation(OnnxRoundTranslation(),
                                      converter_type('Round', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ROUND])


# ------------------------------------------------------------------------------
#   Sigmoid
# ------------------------------------------------------------------------------
class OnnxSigmoidTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sigmoid', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_neuron_type(src_op.op_type), alpha=1.0)


OnnxTranslations.register_translation(OnnxSigmoidTranslation(), converter_type('Sigmoid', 'onnx'))


# ------------------------------------------------------------------------------
#   Sign
# ------------------------------------------------------------------------------
class OnnxSignTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sign', [13])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SIGN)


OnnxTranslations.register_translation(OnnxSignTranslation(),
                                      converter_type('Sign', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SIGN])


# ------------------------------------------------------------------------------
#   Sin
# ------------------------------------------------------------------------------
class OnnxSinTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sin', [7])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SIN)


OnnxTranslations.register_translation(OnnxSinTranslation(),
                                      converter_type('Sin', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SIN])


# ------------------------------------------------------------------------------
#   Softmax
# ------------------------------------------------------------------------------
class OnnxSoftmaxTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softmax', [1, 11, 13])

    def add_op(self, src_op, converter_context, **kwargs):
        graph = converter_context.ir_graph
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        translated_ops = self.extract_parameters(src_op, converter_context)
        if len(translated_ops) == 3:
            # translate_ops = [reshape, softmax, reshape]
            # add pre reshape
            pre_reshape_output_names = [input_names[0] + '_reshape']
            pre_reshape_node = graph.add(translated_ops[0], input_names, pre_reshape_output_names)
            graph.add_src_op_info(pre_reshape_node.op.name, input_names, pre_reshape_output_names)

            # add softmax
            softmax_output_names = [input_names[0] + '_softmax']
            softmax_node = graph.add(translated_ops[1], pre_reshape_node.output_names, softmax_output_names)
            graph.add_src_op_info(softmax_node.op.name, pre_reshape_node.output_names, softmax_output_names)

            # add post reshape
            input_buf = graph.get_buffer(input_names[0])
            post_reshape_node = graph.add(translated_ops[2], softmax_node.output_names, output_names,
                                          axis_formats=[input_buf.axis_format])
            graph.add_src_op_info(post_reshape_node.op.name, softmax_node.output_names, output_names)
            last_node = post_reshape_node
        else:
            # translate_ops = [softmax]
            softmax_node = graph.add(translated_ops[0], input_names, output_names)
            graph.add_src_op_info(softmax_node.op.name, input_names, output_names)
            last_node = softmax_node

        return last_node

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        schema = self.op_schema()
        params = extract_attributes(src_op, schema=schema)
        axis = getattr(params, "axis", 1)

        origin_shape = graph.get_buffer(src_op.input[0]).shape
        if axis == 0 or axis == -len(origin_shape):
            raise ValueError("ERROR: Invalid value {} for axis attribute for {} op".format(axis, src_op.op_type))

        # only insert reshape if input rank larger than 2,
        # the reason is that in such case adding reshape won't influence result
        if schema.version[0] < 13 and len(origin_shape) > 2:
            # for softmax with older version, it will flatten dimension after axis(include) and calculate softmax on it
            shape = [*origin_shape[:axis], np.prod(origin_shape[axis:])]

            # flatten dimension after axis (include)
            pre_reshape_node_name = str(src_op.input[0]) + '_flatten'
            pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_node_name, shape=shape)

            # softmax on axis
            softmax_op = op_adapter.SoftmaxOp(str(src_op.name), axis=axis)

            # reshape to origin shape
            post_reshape_node_name = str(src_op.output[0]) + '_reshape'
            post_reshape_op = op_adapter.ReshapeOp(name=post_reshape_node_name, shape=origin_shape)

            translated_ops = [pre_reshape_op, softmax_op, post_reshape_op]
        else:
            translated_ops = [op_adapter.SoftmaxOp(str(src_op.name), axis=axis)]
        return translated_ops


OnnxTranslations.register_translation(OnnxSoftmaxTranslation(),
                                      converter_type('Softmax', 'onnx'),
                                      op_adapter.SoftmaxOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Softplus
# ------------------------------------------------------------------------------
class OnnxSoftplusTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Softplus', [1])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SOFTPLUS)


OnnxTranslations.register_translation(OnnxSoftplusTranslation(),
                                      converter_type('Softplus', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SOFTPLUS])


# ------------------------------------------------------------------------------
#   Sub
# ------------------------------------------------------------------------------
class OnnxSubTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sub', [1, 6, 7])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT
        self.numpy_op = numpy.subtract


OnnxTranslations.register_translation(OnnxSubTranslation(),
                                      converter_type('Sub', 'onnx'),
                                      op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT])


# ------------------------------------------------------------------------------
#   Sum
# ------------------------------------------------------------------------------
class OnnxSumTranslation(ElementwiseBinaryTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sum', [1, 6, 8])
        self.eltwise_type = ir_graph.QNN_OP_ELEMENT_WISE_ADD
        self.numpy_op = numpy.add


OnnxTranslations.register_translation(OnnxSumTranslation(), converter_type('Sum', 'onnx'))


# ------------------------------------------------------------------------------
#   Sqrt
# ------------------------------------------------------------------------------
class OnnxSqrtTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Sqrt', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.ElementwiseUnaryOp(str(src_op.name), eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SQUARE_ROOT)


OnnxTranslations.register_translation(OnnxSqrtTranslation(),
                                      converter_type('Sqrt', 'onnx'),
                                      op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SQUARE_ROOT])


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class OnnxTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Tanh', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_neuron_type(src_op.op_type),
                                   alpha=1.0,
                                   beta=1.0)


OnnxTranslations.register_translation(OnnxTanhTranslation(),
                                      converter_type('Tanh', 'onnx'))


# ------------------------------------------------------------------------------
#   ScaledTanh
# ------------------------------------------------------------------------------
class OnnxScaledTanhTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('ScaledTanh', [1, 6])

    def extract_parameters(self, src_op, converter_context):
        # these parameters belong to ScaledTanh
        params = extract_attributes(src_op, schema=self.op_schema())
        return op_adapter.NeuronOp(str(src_op.name),
                                   op_adapter.NeuronOp.extract_neuron_type(src_op.op_type),
                                   alpha=params.alpha,
                                   beta=params.beta)


# scaledtanh is removed in ONNX release v1.5.0, add if statement to avoid warning
if packaging.version.Version(onnx.__version__) < packaging.version.Version("1.5.0"):
    OnnxTranslations.register_translation(OnnxScaledTanhTranslation(),
                                          converter_type('ScaledTanh', 'onnx'))


# ------------------------------------------------------------------------------
#   TopK
# ------------------------------------------------------------------------------
class OnnxTopKTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('TopK', [1, 10, 11])

    def extract_parameters(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(src_op, schema=self.op_schema())
        input_names = list(src_op.input)
        input_buf = graph.get_buffer(str(src_op.input[0]))
        input_rank = input_buf.rank()
        input_dims = input_buf.get_buf_dims()

        # extract K as input in versions 10, 11 and as parameter in version 1
        if len(input_names) == 2:
            const_op = self.fetch_constant_op(input_names[1], converter_context)
            log_assert(const_op is not None,
                       "Input tensor {} of node {} could not be extracted.".format(input_names[1], src_op.name))
            k = const_op.tensor.astype(numpy.int64).item(0)
        else:
            k = params.k

        largest = params.largest if 'largest' in params else 1
        sorted = params.sorted if 'sorted' in params else 1
        if largest != 1:
            log_assert(largest != 1, "TopK Op {}, attribute largest not supported", src_op.name)
        if sorted != 1:
            log_assert(sorted != 1, "TopK Op {}, attribute sorted not supported", src_op.name)
        axis = params.axis

        if axis < 0:
            axis += input_rank

        log_assert(input_rank >= 1,
                   code_to_message.get_error_message("ERROR_TOPK_INPUT_TENSOR_RANK")(input_rank))
        if axis != input_rank - 1:
            raise ValueError(
                code_to_message.get_error_message("ERROR_TOPK_UNSUPPORTED_LAYER_PARAM")("axis", axis))

        if k < 0 or input_dims[axis] < k:
            raise ValueError(
                code_to_message.get_error_message("ERROR_TOPK_K_INVALID")(k, input_dims[axis]))

        return op_adapter.TopKOp(src_op.name, k=k)

    def extract_input_names(self, src_op, converter_context):
        return [src_op.input[0]]


OnnxTranslations.register_translation(OnnxTopKTranslation(),
                                      converter_type('TopK', 'onnx'),
                                      op_adapter.TopKOp.TRANSLATION_KEY)
