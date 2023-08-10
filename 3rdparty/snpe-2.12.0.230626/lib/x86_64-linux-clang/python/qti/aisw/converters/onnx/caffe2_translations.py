# ==============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from .onnx_translations import *
from .util import *

from qti.aisw.converters.common import ir_graph

# ------------------------------------------------------------------------------
#   AliasWithName
# ------------------------------------------------------------------------------
class OnnxAliasWithNameTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('is_backward', 'i', 0),
                                                ('name', 's', '')])

        return op_adapter.IdentityOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxAliasWithNameTranslation(),
                                      converter_type('AliasWithName', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   BatchPermutation
# ------------------------------------------------------------------------------
class OnnxBatchPermutationTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.BatchPermutationOp(str(src_op.name))


OnnxTranslations.register_translation(OnnxBatchPermutationTranslation(),
                                      converter_type('BatchPermutation', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   CollectRpnProposals
# ------------------------------------------------------------------------------
class OnnxCollectRpnProposalsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('rpn_max_level', 'i', 6),
                                                ('rpn_min_level', 'i', 2),
                                                ('rpn_post_nms_topN', 'i', 2000)])

        if params.rpn_min_level < 2 or params.rpn_min_level > 6:
            raise ValueError("CollectRpnProposals Op {} only support parameter rpn_min_level in range [2, 6], got {}".format(
                    src_op.name, params.rpn_min_level))

        if params.rpn_max_level < 2 or params.rpn_max_level > 6:
            raise ValueError("CollectRpnProposals Op {} only support parameter rpn_max_level in range [2, 6], got {}".format(
                    src_op.name, params.rpn_max_level))

        if params.rpn_max_level < params.rpn_min_level:
            raise ValueError("CollectRpnProposals Op {} expected parameter rpn_max_level >= rpn_min_level, got rpn_max_level {} and rpn_min_level {}".format(
                    src_op.name, params.rpn_max_level, params.rpn_min_level))

        return op_adapter.CollectRpnProposalsOp(str(src_op.name),
                                                rpn_min_level=params.rpn_min_level,
                                                rpn_max_level=params.rpn_max_level,
                                                post_nms_top=params.rpn_post_nms_topN)


OnnxTranslations.register_translation(OnnxCollectRpnProposalsTranslation(),
                                      converter_type('CollectRpnProposals', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   DistributeFpnProposals
# ------------------------------------------------------------------------------
class OnnxDistributeFpnProposalsTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('legacy_plus_one', 'i'),
                                                ('roi_canonical_level', 'i', 4),
                                                ('roi_canonical_scale', 'i', 224),
                                                ('roi_max_level', 'i', 5),
                                                ('roi_min_level', 'i', 2)])

        if params.legacy_plus_one != 0:
            raise ValueError(
                "The parameter 'legacy_plus_one' in DistributeFpnProposals op do not support non-zero values")

        if params.roi_min_level < 2 or params.roi_min_level > 5:
            raise ValueError(
                "The parameter 'roi_min_level' in DistributeFpnProposals op must be in range [2,5]")

        if params.roi_max_level < 2 or params.roi_max_level > 5:
            raise ValueError(
                "The parameter 'roi_max_level' in DistributeFpnProposals op must be in range [2,5]")

        if params.roi_max_level < params.roi_min_level:
            raise ValueError("The parameter 'roi_max_level' must be >= 'roi_min_level' in DistributeFpnProposals op")

        return op_adapter.DistributeFpnProposalsOp(str(src_op.name),
                                                   roi_canonical_level=params.roi_canonical_level,
                                                   roi_canonical_scale=params.roi_canonical_scale,
                                                   roi_max_level=params.roi_max_level,
                                                   roi_min_level=params.roi_min_level)

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        # change output order of Caffe2 to align with QNN opdef.
        output_names.insert(0, output_names.pop())
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        return node


OnnxTranslations.register_translation(OnnxDistributeFpnProposalsTranslation(),
                                      converter_type('DistributeFpnProposals', 'onnx_caffe2'))


# ------------------------------------------------------------------------------
#   ResizeNearest
# ------------------------------------------------------------------------------
class OnnxResizeNearestTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, converter_context):
        params = extract_attributes(src_op,
                                    attr_infos=[('height_scale', 'f', 1.0),
                                                ('width_scale', 'f', 1.0),
                                                ('order', 's', '')])

        return op_adapter.ResizeOp(str(src_op.name),
                                   interpolation_mode=ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
                                   scale_height=params.height_scale,
                                   scale_width=params.width_scale)


OnnxTranslations.register_translation(OnnxResizeNearestTranslation(),
                                      converter_type('ResizeNearest', 'onnx_caffe2'))
