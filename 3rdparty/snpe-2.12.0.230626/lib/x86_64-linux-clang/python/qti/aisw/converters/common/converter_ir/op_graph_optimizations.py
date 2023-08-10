# ==============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import numpy as np
from functools import reduce

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import translation, op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings, InputLayout
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrder, CaffeAxisOrder, SpatialLastAxisOrder, RelayAxisOrder
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import code_to_message, translation_utils


# ------------------------------
#   Module Level enum/Functions
# ------------------------------
INJECT_CAST_FOR_GATHER = "INJECT_CAST_FOR_GATHER"
REMOVE_IDENTITY = "REMOVE_IDENTITY"
REMOVE_CAST_IDENTITY = "REMOVE_CAST_IDENTITY"
REMOVE_DISCONNECTED = "REMOVE_DISCONNECTED"
MATCH_CHANNELSHUFFLE = "MATCH_CHANNELSHUFFLE"
MATCH_GATHERND = "MATCH_GATHERND"
MATCH_GELU = "MATCH_GELU"
MATCH_HARDSWISH = "MATCH_HARDSWISH"
MATCH_LAYERNORM = "MATCH_LAYERNORM"
MATCH_CAFFE_SSD_TO_TF = "MATCH_CAFFE_SSD_TO_TF"
MATCH_SPACETODEPTH = "MATCH_SPACETODEPTH"
MATCH_DEPTHTOSPACE = "MATCH_DEPTHTOSPACE"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
SQUASH_BOX_DECODER = "SQUASH_BOX_DECODER"
SQUASH_SUM = "SQUASH_SUM"
SQUASH_PROD = "SQUASH_PROD"
SQUASH_DIV = "SQUASH_DIV"
SQUASH_SUB = "SQUASH_SUB"
SQUASH_PAD = "SQUASH_PAD"
SQUASH_RESHAPE = "SQUASH_RESHAPE"
FOLD_CAST = "FOLD_CAST"
FOLD_CONCATS = "FOLD_CONCATS"
FOLD_RESHAPES = "FOLD_RESHAPES"
AXES_TO_SPATIAL_FIRST_ORDER = "AXES_TO_SPATIAL_FIRST_ORDER"
ADD_QPARAMS = "ADD_QPARAMS"
ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE = "ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE"
ADJUST_NMS_FEATURE_DIMS = "ADJUST_NMS_FEATURE_DIMS"
EXTRACT_COLOR_TRANSFROM = "EXTRACT_COLOR_TRANSFROM"
OPTIMIZE_NEG = "OPTIMIZE_NEG"
PREPROCESS_ROI_POOL_INPUTS = "PREPROCESS_ROI_POOL_INPUTS"
UNROLL_LSTM_TIME_STEPS = "UNROLL_LSTM_TIME_STEPS"
EXPAND_LSTM_OP_STRUCTURE = "EXPAND_LSTM_OP_STRUCTURE"
UNROLL_GRU_TIME_STEPS = "UNROLL_GRU_TIME_STEPS"
EXPAND_GRU_OP_STRUCTURE = "EXPAND_GRU_OP_STRUCTURE"
MERGE_LOW_LEVEL_OPS_TO_LAYERS = "MERGE_LOW_LEVEL_OPS_TO_LAYERS"
REMOVE_QUANT_NODES = "REMOVE_QUANT_NODES"
SQUASH_QUANT_NODES = "SQUASH_QUANT_NODES"
ALIGN_MATMUL_RANKS = "ALIGN_MATMUL_RANKS"
PREPARE_INPUTS_AS_PARAMS = "PREPARE_INPUTS_AS_PARAMS"
HANDLE_GATHER_NEGATIVE_INDICES = "HANDLE_GATHER_NEGATIVE_INDICES"
PREPARE_BIASES = "PREPARE_BIASES"
expand_1d_spatial_nn_nodes = "expand_1d_spatial_nn_nodes"
supported_opt_list = [SQUASH_SCALE, SQUASH_PROD, SQUASH_DIV, SQUASH_SUM, SQUASH_SUB, SQUASH_RESHAPE, SQUASH_BATCHNORM, FOLD_CAST, FOLD_CONCATS, FOLD_RESHAPES,
                      MATCH_CHANNELSHUFFLE, MATCH_GATHERND, MATCH_GELU, MATCH_HARDSWISH, MATCH_LAYERNORM, AXES_TO_SPATIAL_FIRST_ORDER,
                      REMOVE_IDENTITY, REMOVE_CAST_IDENTITY, ADD_QPARAMS, ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE, MATCH_DEPTHTOSPACE,
                      ADJUST_NMS_FEATURE_DIMS, EXTRACT_COLOR_TRANSFROM, OPTIMIZE_NEG, MATCH_SPACETODEPTH,
                      PREPROCESS_ROI_POOL_INPUTS, UNROLL_LSTM_TIME_STEPS, EXPAND_LSTM_OP_STRUCTURE, SQUASH_PAD,
                      MERGE_LOW_LEVEL_OPS_TO_LAYERS, INJECT_CAST_FOR_GATHER, REMOVE_QUANT_NODES, SQUASH_QUANT_NODES,
                      ALIGN_MATMUL_RANKS, PREPARE_INPUTS_AS_PARAMS, HANDLE_GATHER_NEGATIVE_INDICES, PREPARE_BIASES,
                      expand_1d_spatial_nn_nodes, UNROLL_GRU_TIME_STEPS, EXPAND_GRU_OP_STRUCTURE]

spatial_first_format_to_channel_first_permute_order = {'NDHWC': AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                                       'NSC': AxisTracker.AxisFormat.NSC_TO_NCS,
                                                       'NFC': AxisTracker.AxisFormat.NFC_TO_NCF,
                                                       'NTF': AxisTracker.AxisFormat.NTF_TO_TNF}
spatial_first_format_to_channel_first_format = {'NDHWC': AxisTracker.AxisFormat.NCDHW,
                                                'NSC': AxisTracker.AxisFormat.NCS,
                                                'NFC': AxisTracker.AxisFormat.NCF,
                                                'NTF': AxisTracker.AxisFormat.TNF}
OptimizationTranslations = translation.TranslationBank()


class IROptimizations(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(IROptimizations.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument("--dumpIR", action="store_true",
                                       help=argparse.SUPPRESS,
                                       default=False)
            self.add_optional_argument("--disable_batchnorm_folding",
                                       default=False,
                                       action="store_true")
            self.add_optional_argument("--squash_box_decoder",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--match_caffe_ssd_to_tf",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--adjust_nms_features_dims",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--extract_color_transform",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--preprocess_roi_pool_inputs",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_axes_to_spatial_first_order",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_lstm_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_lstm_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_gru_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_gru_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--force_prune_cast_ops",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--inject_cast_for_gather",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--use_convert_quantization_nodes",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--align_matmul_ranks",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--prepare_inputs_as_params",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--handle_gather_negative_indices",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--enable_match_gathernd",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--keep_disconnected_nodes",
                                       default=False,
                                       help="Disable Optimization that removes Ops not connected to the main graph.\n"
                                            "This optimization uses output names provided over commandline OR\n"
                                            "inputs/outputs extracted from the Source model to determine the main graph",
                                       action="store_true")

    def __init__(self, args):
        self.dump_ir_graph = args.dumpIR
        self.enable_batchnorm_folding = not args.disable_batchnorm_folding
        self.squash_box_decoder = args.squash_box_decoder
        self.match_caffe_ssd_to_tf = args.match_caffe_ssd_to_tf
        self.adjust_nms_features_dims = args.adjust_nms_features_dims
        self.extract_color_transform = args.extract_color_transform
        self.perform_axes_to_spatial_first_order = args.perform_axes_to_spatial_first_order
        self.preprocess_roi_pool_inputs = args.preprocess_roi_pool_inputs
        self.unroll_lstm_time_steps = args.unroll_lstm_time_steps
        self.expand_lstm_op_structure = args.expand_lstm_op_structure
        self.unroll_gru_time_steps = args.unroll_gru_time_steps
        self.expand_gru_op_structure = args.expand_gru_op_structure
        self.force_prune_cast_ops = args.force_prune_cast_ops
        self.inject_cast_for_gather = args.inject_cast_for_gather
        self.use_convert_quantization_nodes = args.use_convert_quantization_nodes
        self.align_matmul_ranks = args.align_matmul_ranks
        self.prepare_inputs_as_params = args.prepare_inputs_as_params
        self.handle_gather_negative_indices = args.handle_gather_negative_indices
        self.enable_match_gathernd = args.enable_match_gathernd
        self.keep_disconnected_nodes = args.keep_disconnected_nodes

    def optimize(self, graph):
        # apply graph transformations
        log_debug2("Applying graph Optimizations...")

        # Dump the IR for debug before or after an optimization using graph.dump_json(<filename>)
        if self.dump_ir_graph:
            log_info("Dumping IR graph before all optimizations as IRGraph_before_optimizations.json")
            graph.dump_json("IRGraph_before_optimizations.json")

        # A dict containing the IO tensor name and corresponding layout obtained from the original model.
        # This information is taken from the intial unoptimized IR graph. Hence, this line of code should
        # be before any optimization.
        original_io_layouts = self.get_original_io_layouts(graph) if graph.preserve_io_layout_passed else {}

        # Remove nodes disconnected from the main graph
        # This function should be in the beginning and the end.
        if not self.keep_disconnected_nodes:
            remove_disconnected_nodes(graph)

        # First attempt to match and fold quant nodes, then remove any remaining
        if graph.keep_quant_nodes:
            if self.use_convert_quantization_nodes:
                OptimizationTranslations.apply_method_to_graph(SQUASH_QUANT_NODES, graph, fail_if_no_method=False)
        else:
            OptimizationTranslations.apply_method_to_all_ops(REMOVE_QUANT_NODES, graph, fail_if_no_method=False)

        if graph.has_user_quantization_overrides():
            self.populate_quantization_params(graph)

        # TODO Remove this preparation once backends are able to consume optional bias tensors
        # prepares bias tensors from frontends for consumption by optimizations and backends
        OptimizationTranslations.apply_method_to_all_ops(PREPARE_BIASES, graph, fail_if_no_method=False)

        # this optimization needs to be run first before any other optimizations
        OptimizationTranslations.apply_method_to_graph(MERGE_LOW_LEVEL_OPS_TO_LAYERS, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PAD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_CAST, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_CONCATS, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_RESHAPES, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_CHANNELSHUFFLE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_GELU, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_HARDSWISH, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_LAYERNORM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_SPACETODEPTH, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_DEPTHTOSPACE, graph, fail_if_no_method=False)

        if self.enable_match_gathernd:
            OptimizationTranslations.apply_method_to_graph(MATCH_GATHERND, graph, fail_if_no_method=False)

        # Element-wise squashing optimizations. This shall be done after matching larger sequences as they single-op
        # squashing into previous layer
        OptimizationTranslations.apply_method_to_graph(SQUASH_SCALE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PROD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_DIV, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUB, graph, fail_if_no_method=False)

        if self.enable_batchnorm_folding:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)
        if self.squash_box_decoder:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BOX_DECODER, graph, fail_if_no_method=False)
        if self.match_caffe_ssd_to_tf:
            OptimizationTranslations.apply_method_to_graph(MATCH_CAFFE_SSD_TO_TF, graph, fail_if_no_method=False)
        if self.adjust_nms_features_dims:
            OptimizationTranslations.apply_method_to_graph(ADJUST_NMS_FEATURE_DIMS, graph, fail_if_no_method=False)
        if self.extract_color_transform:
            OptimizationTranslations.apply_method_to_graph(EXTRACT_COLOR_TRANSFROM, graph, fail_if_no_method=False)

        OptimizationTranslations.apply_method_to_graph(SQUASH_RESHAPE, graph, fail_if_no_method=False)
        # ------------------------------------------------------------------------------
        #   PRE-PROCESSING
        # TODO: Move once optimizations are split into backend specific sections
        # ------------------------------------------------------------------------------
        # pre-process roi inputs
        if self.preprocess_roi_pool_inputs:
            OptimizationTranslations.apply_method_to_graph(PREPROCESS_ROI_POOL_INPUTS, graph, fail_if_no_method=False)

        # Performs pruning of cast Ops that are identity, if force_prune is set then all cast ops are pruned
        # TODO: remove separate identity call for casts when Cast supported by all backends
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_CAST_IDENTITY, graph, force_prune=self.force_prune_cast_ops,
                                                         fail_if_no_method=False)

        OptimizationTranslations.apply_method_to_all_ops(expand_1d_spatial_nn_nodes, graph, fail_if_no_method=False)

        # Ensure matmul dims are handled/squashed as needed.
        if self.align_matmul_ranks:
            OptimizationTranslations.apply_method_to_all_ops(ALIGN_MATMUL_RANKS, graph, fail_if_no_method=False)

        if self.dump_ir_graph and get_log_level() == logging.VERBOSE:
            graph.dump_json("IRGraph_before_layout_change.json")

        # Save source axis formats
        graph.save_src_axis_formats()

        # transition to NSC
        if self.perform_axes_to_spatial_first_order:
            OptimizationTranslations.apply_method_to_all_ops(AXES_TO_SPATIAL_FIRST_ORDER, graph)

        if self.dump_ir_graph and get_log_level() == logging.VERBOSE:
            graph.dump_json("IRGraph_after_layout_change.json")

        self.squash_multiple_permute(graph)

        if self.dump_ir_graph and get_log_level() == logging.VERBOSE:
            graph.dump_json("IRGraph_after_removing_multiple_permutes.json")

        # Optimize negations which typically apply to binary eltwise operations.
        OptimizationTranslations.apply_method_to_graph(OPTIMIZE_NEG, graph, fail_if_no_method=False)

        # add transpose after output reshape
        OptimizationTranslations.apply_method_to_all_ops(ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE, graph, fail_if_no_method=False)

        # remove IDENTITYs, which may include trivial permutes at this point
        # This may happen because some ops result in constant attributes that are absorbed by the layers
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_IDENTITY, graph, fail_if_no_method=False)

        # this is to squash batchnorm for the case [FC, Reshape, BN] --> [FC, Reshape]
        if self.enable_batchnorm_folding:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)

        # add op-specific quantization encodings to QParams Record.
        OptimizationTranslations.apply_method_to_all_ops(ADD_QPARAMS, graph, fail_if_no_method=False)

        # Apply unrolling to replace RolledLstmOp with LstmOp, and perform pre-processing
        if self.unroll_lstm_time_steps:
            OptimizationTranslations.apply_method_to_graph(UNROLL_LSTM_TIME_STEPS, graph, fail_if_no_method=False)

        # Apply expansion to LSTM op
        if self.expand_lstm_op_structure:
            OptimizationTranslations.apply_method_to_graph(EXPAND_LSTM_OP_STRUCTURE, graph, fail_if_no_method=False)

        # Apply time-unrolling to GRU op
        if self.unroll_gru_time_steps:
            OptimizationTranslations.apply_method_to_graph(UNROLL_GRU_TIME_STEPS, graph, fail_if_no_method=False)

        # Apply expansion to GRU op
        if self.expand_gru_op_structure:
            OptimizationTranslations.apply_method_to_graph(EXPAND_GRU_OP_STRUCTURE, graph, fail_if_no_method=False)

        # Pre-processing of gather indices input
        if self.handle_gather_negative_indices:
            OptimizationTranslations.apply_method_to_all_ops(HANDLE_GATHER_NEGATIVE_INDICES, graph,
                                                             fail_if_no_method=False)

        # TODO Remove optimization once casts are properly removed in optimization stage
        if self.inject_cast_for_gather:
            OptimizationTranslations.apply_method_to_all_ops(INJECT_CAST_FOR_GATHER, graph, fail_if_no_method=False)

        # Prepares inputs in converter IR as parameters, as needed
        if self.prepare_inputs_as_params:
            OptimizationTranslations.apply_method_to_all_ops(PREPARE_INPUTS_AS_PARAMS, graph, fail_if_no_method=False)

        if graph.preserve_io_layout_passed:
            if self.dump_ir_graph:
                log_info("Dumping IR graph before applying preserve IO changes as IRGraph_before_preserveIO.json")
                graph.dump_json("IRGraph_before_preserveIO.json")
            self.preserve_io_layout(graph, original_io_layouts)

        if graph.user_custom_io:
            self.validate_custom_io_config(graph)
            if self.dump_ir_graph:
                log_info("Dumping IR graph before custom IO as IRGraph_before_customIO.json")
                graph.dump_json("IRGraph_before_customIO.json")

            self.apply_custom_io_change(graph)
            self.populate_custom_io_quantization_params(graph)

        # Remove nodes disconnected from the main graph
        # This function should be in the beginning and the end.
        if not self.keep_disconnected_nodes:
            remove_disconnected_nodes(graph)

        if self.dump_ir_graph:
            log_info("Dumping IR graph after all optimizations as IRGraph_after_optimizations.json")
            graph.dump_json("IRGraph_after_optimizations.json")

        # re-evaluate graph macs and params_count given optimization might have added/removed certain ops
        graph.reeval_macs_params()

        return graph

    def squash_multiple_permute(self, graph):
        def is_valid_sequence(nodes_tuple):
            nonlocal graph
            first_permute, second_permute = nodes_tuple
            first_permute_output_nodes = graph.get_op_output_nodes(first_permute)
            if len(first_permute_output_nodes) > 1 or first_permute_output_nodes[0] != second_permute:
                return False
            return True

        sequence = [
                    ("Transpose",
                        (),
                        ()
                    ),
                    ("Transpose",
                        ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
                        ()
                    )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_sequence)

        for node_tuple in matched_node_list:
            first_permute, second_permute = node_tuple

            first_permute_order = first_permute.op.perm
            second_permute_order = second_permute.op.perm

            new_order = first_permute_order[:]
            for i, val in enumerate(second_permute_order):
                new_order[i] = first_permute_order[val]

            graph.squash(second_permute, first_permute.output_names[0])
            first_permute.op.perm = new_order
            if new_order == list(range(len(new_order))):
                graph.squash(first_permute, first_permute.input_names[0])

            log_debug2("Found Multiple Permute: First Permute={}, Second Permute={}, First Order={}, "
                       "Second Order={}, Final Order={}".format(first_permute.op.name,
                                                                second_permute.op.name,
                                                                first_permute_order,
                                                                second_permute_order,
                                                                new_order))

    @staticmethod
    def get_default_enc(name, enc):
        # Everything is optional except bw. Default to 0s and overwrite with actual or calculated values later
        return {"name": name,
                "min": numpy.zeros(len(enc), numpy.float32),
                "max": numpy.zeros(len(enc), numpy.float32),
                "bw": int(enc[0]['bitwidth']),
                "offset": numpy.zeros(len(enc), numpy.int32),
                "scale": numpy.zeros(len(enc), numpy.float32),
                "is_symmetric": enc[0]['is_symmetric'].lower() == "true" if 'is_symmetric' in enc[0] else False,
                "is_fixed_point": enc[0]['dtype'].lower() == "int" if 'dtype' in enc[0] else True,
                "overridden": True}

    @staticmethod
    def extract_encoding_dict(name, enc):

        # Grab a default encoding
        new_enc = IROptimizations.get_default_enc(name, enc)
        # Loop through each encoding for the tensor. More than one indicates the tensor has
        # per-channel (axis) encodings
        for idx, e in enumerate(enc):
            try:
                if 'bitwidth' in e:
                    log_assert(e['bitwidth'] == new_enc['bw'],
                               'Mismatching bitwidths {} and {} for encoding {}',
                               e['bitwidth'], new_enc['bw'], name)
                if 'is_symmetric' in e:
                    log_assert((e['is_symmetric'].lower() == "true") == new_enc['is_symmetric'],
                               'Encodings {} for tensor {} cannot be a mix of symmetric and asymmetric',
                               enc, name)

                if 'is_fixed_point' in e:
                    log_assert((e['is_fixed_point'].lower() == "true") == new_enc['is_fixed_point'],
                               'Encodings {} for tensor {} cannot be a mix of fixed and floating point',
                                enc, name)
                # For min/max and scale/offset if either of the pairs is provided both must be or throw
                if any(k in ['min','max'] for k in e.keys()):
                    new_enc['min'][idx] = float(e["min"])
                    new_enc['max'][idx] = float(e["max"])
                if any(k in ['scale','offset'] for k in e.keys()):
                    new_enc['scale'][idx] = float(e["scale"])
                    new_enc['offset'][idx] = int(-abs(e['offset']))

                # User quantization overrides may specify only scale/offset/bitwidth and then min/max can be calculated
                symmetric_max = (2 ** (int(new_enc['bw']) - 1))

                if all(key not in e for key in ['min', 'max']) \
                        and all(key in e for key in ['scale']):
                    if new_enc['is_symmetric']:
                        new_enc['min'][idx] = (-symmetric_max + 1) * e['scale']
                        new_enc['max'][idx] = (symmetric_max - 1) * e['scale']
                    else:
                        new_enc['min'][idx] = new_enc['offset'][idx] * e['scale']
                        new_enc['max'][idx] = (((2 ** e['bitwidth']) - 1) + new_enc['offset'][idx]) * e['scale']

                # Symmetric weights should have 0 offset overridden with -symmetric_max, or already be equal to -symmetrich_max
                if new_enc['is_symmetric']:
                    if new_enc['offset'][idx] == 0:
                        new_enc['offset'][idx] = -symmetric_max
                    else:
                        if new_enc['offset'][idx] != -symmetric_max:
                            raise ValueError("Invalid offset overridden for symmetric encodings got {}, expected {}."
                                             .format(new_enc['offset'], -symmetric_max))
            except Exception as exc:
                log_error("Error: {} in tensor {} encoding {}. Min/max or scale/offset pairs must be present together.".format(str(exc), name, enc))
                raise exc
        # Force the axis to default of 3 for axis quant
        if len(enc) > 1:
            new_enc['axis'] = 3

        return new_enc

    def populate_quantization_params(self, ir_graph):

        def _adjust_bias_encoding(ir_graph):
            # The bias encoding in ir_graph.quantization_params corresponds to BiasAdd node as weights, we need to alter the name
            # 'weights' with 'bias' and add it to the params_encodings of the conv, deconv, matmul or fc node prior to the BiasAdd
            # so that the quantizer can get the bias encoding properly.
            for node in ir_graph.list_nodes():
                if node.op.hasattr('bias_op_name'):
                    _bias_op_name = node.op.bias_op_name

                    if _bias_op_name and _bias_op_name in ir_graph.quantization_params:
                        param_encodings = ir_graph.get_layer_quantization_param(_bias_op_name)[op_graph.QuantParams.PARAM_ENCODINGS]
                        if len(param_encodings) > 0:
                           _bias_encoding = param_encodings[0]
                           _bias_encoding['name'] = 'bias' # alter name 'weights' with 'bias'
                           ir_graph.add_quantization_params(node.op.name, param_encodings=_bias_encoding)

        q = ir_graph.user_quantization_overrides
        acts = q['activation_encodings']
        params = q['param_encodings']
        encoding_count = 0

        # Graph inputs are special cases because they aren't owned by a node until IR conversion
        inputs = ir_graph.get_input_nodes_to_graph()
        for i in inputs:
            n = i.op.name
            if n in acts:
                encoding_count += 1
                ir_graph.add_quantization_params(n, output_encodings=[IROptimizations.extract_encoding_dict(n, acts[n])])

        # Walk through the original source framework op->input mapping to find the weights
        for op_name, op in ir_graph.src_graph_op_info.items():
            param_encs = []

            inputs = op['inputs']
            node = None
            if op_name in ir_graph.nodes_by_name:
                node = ir_graph.nodes_by_name[op_name]
            if inputs:
                for idx, i in enumerate(inputs):
                    if i in params:
                        encoding_count += 1
                        # If this encoding name is bias op name, the name should be set be "bias"
                        if node is not None and node.op.hasattr('bias_op_name') and node.op.bias_op_name == i:
                            param_encs.append(IROptimizations.extract_encoding_dict('bias', params[i]))
                        else:
                            param_encs.append(IROptimizations.extract_encoding_dict('weights', params[i]))
                # only add quantization params if param_encs is not empty
                if param_encs:
                    ir_graph.add_quantization_params(op_name, param_encodings=param_encs)

        # adjust the bias encoding for 'fully_connected', 'convolution', 'TransposeConv2d' ops.
        _adjust_bias_encoding(ir_graph)

        # Walk through the activations and lookup in the IR graph since folding, squashing, pruning
        # may have moved the activation names to new ops.
        for act in acts:
            act_encs = []
            if ir_graph.has_buffer(act):
                op = ir_graph.get_producer_op(act)
                encoding_count += 1
                act_encs.append(IROptimizations.extract_encoding_dict(act, acts[act]))
                ir_graph.add_quantization_params(op.name, output_encodings=act_encs)

        log_info('Processed '+ str(encoding_count)+' quantization encodings')

    def populate_custom_io_quantization_params(self, ir_graph):
        """
        Populates the quantization_params of the ir_graph with the scale and offset provided in the custom IO YAML file.

        :param graph: an IROpgraph object
        """
        def custom_io_to_quant_enc(entry):
            # Populates the 'enc' dictionary with the data from the custom IO YAML file.
            # The format of 'enc' dictionary is similar to the one generated from quantization_overrides json file.
            datatype_to_bw = {
                'float32':32,
                'float16':16,
                'int32':32,
                'uint32':32,
                'int8':8,
                'uint8':8,
            }
            datatype_to_range = {
                'int8':(-128,127),
                'uint8':(0,255)
            }
            scale = entry['QuantParam']['Scale']
            offset = entry['QuantParam']['Offset']
            # Default datatype for quantized inputs is uint8 in case of custom IO.
            # If 'QuantParam' are provided and no 'Datatype' is provided, it is assumed to be uint8.
            custom_datatype = 'uint8'
            if 'Datatype' in entry:
                custom_datatype = entry['Datatype']
            minVal = scale*(datatype_to_range[custom_datatype][0] + offset) if custom_datatype in datatype_to_range else 0.0
            maxVal = scale*(datatype_to_range[custom_datatype][1] + offset) if custom_datatype in datatype_to_range else 0.0
            isSymmetricType = 'True' if custom_datatype == 'int8' else 'False'
            enc = {
                'bitwidth':datatype_to_bw[entry['Datatype']],
                'scale':entry['QuantParam']['Scale'],
                'offset':entry['QuantParam']['Offset'],
                'min':minVal,
                'max':maxVal,
                'is_symmetric':isSymmetricType
            }
            return [enc]

        for entry in ir_graph.user_custom_io:
            if "QuantParam" in entry:
                buffer_name = str(entry['IOName'])
                enc = custom_io_to_quant_enc(entry)
                isInput = False

                # Graph inputs are special cases because they aren't owned by a node until IR conversion
                inputs = ir_graph.get_input_nodes_to_graph()
                for i in inputs:
                    n = i.op.name
                    if n == buffer_name:
                        ir_graph.add_quantization_params(n, output_encodings=[IROptimizations.extract_encoding_dict(n, enc)])
                        isInput = True

                # Walk through the activations and lookup in the IR graph since folding, squashing, pruning
                # may have moved the activation names to new ops.
                if not isInput and ir_graph.has_buffer(buffer_name):
                    op = ir_graph.get_producer_op(buffer_name)
                    ir_graph.add_quantization_params(op.name, output_encodings=[IROptimizations.extract_encoding_dict(buffer_name, enc)])

    # A method to get the layout of the inputs and outputs in the graph.
    def get_original_io_layouts(self, graph):
        original_io_layouts = {}
        for node in graph.get_input_nodes_to_graph() + graph.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                original_io_layouts[buffer_name] = graph.buffers[buffer_name].axis_format
        return original_io_layouts

    # A common method to modify the IO layout for the --preserve_io and --custom_io option.
    def modify_io_layout(self, graph, buffer_name, initial_axis_format, final_axis_format, isInput = True):
        if graph.buffers[buffer_name].rank() <= 2:
            return
        if initial_axis_format == final_axis_format:
            return
        permute_order, reverse_order = self.get_io_permute_order(initial_axis_format, final_axis_format)
        log_assert((permute_order is not None and reverse_order is not None),"Invalid layout tranformation for buffer {}."
            .format(buffer_name))
        if isInput:
            # For modifying the layout of input tensors, modify the shape of the input node
            # and insert a permute op to get the data in the layout that the graph expects
            node = graph.buffers[buffer_name].producer
            new_shape = [0]*(len(reverse_order))
            for i in range(len(reverse_order)):
                new_shape[i] = node.op.shape[reverse_order[i]]
            buf = graph.get_buffer(buffer_name)
            buf.shape = new_shape
            buf.axis_format = final_axis_format
            node.op.shape = buf.shape
            # Int64 inputs present a special case while preserving both layout and datatype. In this case, the following
            # sequence of Input(int64) -> Transpose -> Cast(to int32) does not work while quantizing the network as
            # CPU backend does not allow int64 inputs for Transpose Op. Hence, the following sequence is created instead:
            # Input(int64) -> Cast (to int32) -> Transpose
            if (buffer_name in graph.preserve_datatype_tensors and graph.preserve_datatype_tensors[buffer_name] == 'int64') or\
                (graph.keep_int64_inputs and node.op.input_dtype == np.dtype('int64')):
                for consumer in graph.buffers[buffer_name].consumers:
                    if consumer.op.name == buffer_name + '_cast_int32':
                        for output_buffer in consumer.output_names:
                            buffer = graph.buffers[output_buffer]
                            buffer.shape = new_shape
                            buffer.axis_format = final_axis_format
                            consumers = [str(name) for name in graph.buffers[buffer.name].consumers]
                            graph.inject_implicit_permute(buffer.name, initial_axis_format, permute_order, consumers)
            else:
                consumers = [str(name) for name in graph.buffers[buffer_name].consumers]
                graph.inject_implicit_permute(buffer_name, initial_axis_format, permute_order, consumers)
        else:
            # For modifying the layout of output tensors, inject a permute op after the output and
            # modify the buffer names and connections appropriately so that the output names
            # are same as in the original graph.
            new_output_node = graph.inject_implicit_permute(buffer_name, final_axis_format, reverse_order, [])
            new_buffer_name = graph.get_implicit_permute_node_name(buffer_name, initial_axis_format)
            new_output_buffer = graph.get_buffer(new_output_node.output_names[0])
            del graph.buffers[new_output_node.output_names[0]]
            graph.naming_policy.remove_output_name(new_output_node.output_names[0])
            old_output_buffer = graph.get_buffer(buffer_name)
            original_output_node = old_output_buffer.producer

            # Update the name of the buffer (which was originally the output but now is consumed by the permute node)
            old_output_buffer.name = new_buffer_name
            # Map the buffer to the correct name in the graph.buffers dictionary
            graph.buffers[new_buffer_name] = old_output_buffer
            # Change the name of the new output buffer to the original output buffer name
            new_output_buffer.name = buffer_name
            # Map the buffer to the correct name in the graph.buffers dictionary
            graph.buffers[buffer_name] = new_output_buffer
            # Make appropriate changes in the connections between nodes.
            # Update the consumer nodes.
            for consumer in old_output_buffer.consumers:
                if consumer.op.name == new_output_node.op.name:
                    continue
                in_idx = consumer.input_names.index(buffer_name)
                consumer.input_names[in_idx] = new_buffer_name

            # Update the producer nodes.
            in_idx = new_output_node.input_names.index(buffer_name)
            new_output_node.input_names[in_idx] = new_buffer_name
            new_output_node.output_names[0] = buffer_name
            out_idx = original_output_node.output_names.index(buffer_name)
            original_output_node.output_names[out_idx] = new_buffer_name

            # Update the new buffer name in quantization_params dictionary for the original output node
            if original_output_node.op.name in graph.quantization_params.keys():
                for encoding_dict in graph.quantization_params[original_output_node.op.name]['output_encodings']:
                    if encoding_dict['name'] == buffer_name:
                        encoding_dict['name'] = new_buffer_name

    def preserve_io_layout(self, graph, original_io_layouts):
        if graph.preserve_io_layout_passed == 1:
            for arg in graph.preserve_io:
                if arg[0] == 'layout':
                    for buffer_name in arg[1:]:
                        graph.preserve_layout_tensors.add(buffer_name)
        elif graph.preserve_io_layout_passed == 2:
            # If the user has passed just the --preserve_io without listing tensor names then preserve the
            # layout for all IO tensors except those tensors whose layout is set using the --custom_io option
            for node in graph.get_input_nodes_to_graph() + graph.get_output_nodes_of_graph():
                for buffer_name in node.output_names:
                    graph.preserve_layout_tensors.add(buffer_name)

        # Skipping those IO tensors whose layout is set using the --custom_io option
        graph.preserve_layout_tensors = graph.preserve_layout_tensors - set(graph.custom_io_axis_formats)

        for node in graph.get_input_nodes_to_graph():
            for buffer_name in node.output_names:
                if buffer_name in graph.preserve_layout_tensors:
                    initial_axis_format = graph.buffers[buffer_name].axis_format
                    if initial_axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                        # Skip preserving layout for tensors whose layout is NONTRIVIAL
                        # NOTE: If the Op at input enforces an axis format (eg: convolution op) and the user passes
                        # NONTRIVIAL for such an input, then it's layout cannot be preserved using the preserve_io option.
                        continue
                    preserve_axis_format = original_io_layouts[buffer_name]
                    self.modify_io_layout(graph, buffer_name, initial_axis_format, preserve_axis_format, isInput = True)
        for node in graph.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                if buffer_name in graph.preserve_layout_tensors:
                    initial_axis_format = graph.buffers[buffer_name].axis_format
                    if initial_axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                        # Skip preserving layout for tensors whose layout is NONTRIVIAL
                        continue
                    preserve_axis_format = original_io_layouts[buffer_name]
                    self.modify_io_layout(graph, buffer_name, initial_axis_format, preserve_axis_format, isInput = False)

    def get_io_permute_order(self, initial_axis_format, custom_axis_format):
        permute_order = None
        reverse_order = None
        if custom_axis_format == AxisTracker.AxisFormat.NCDHW and initial_axis_format== AxisTracker.AxisFormat.NDHWC :
            permute_order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            reverse_order = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
        elif custom_axis_format == AxisTracker.AxisFormat.NDHWC and initial_axis_format== AxisTracker.AxisFormat.NCDHW :
            permute_order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            reverse_order = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
        elif custom_axis_format == AxisTracker.AxisFormat.NCS and initial_axis_format== AxisTracker.AxisFormat.NSC :
            permute_order = AxisTracker.AxisFormat.NCS_TO_NSC
            reverse_order = AxisTracker.AxisFormat.NSC_TO_NCS
        elif custom_axis_format == AxisTracker.AxisFormat.NSC and initial_axis_format== AxisTracker.AxisFormat.NCS :
            permute_order = AxisTracker.AxisFormat.NSC_TO_NCS
            reverse_order = AxisTracker.AxisFormat.NCS_TO_NSC
        elif custom_axis_format == AxisTracker.AxisFormat.NCF and initial_axis_format== AxisTracker.AxisFormat.NFC :
            permute_order = AxisTracker.AxisFormat.NCF_TO_NFC
            reverse_order = AxisTracker.AxisFormat.NFC_TO_NCF
        elif custom_axis_format == AxisTracker.AxisFormat.NFC and initial_axis_format== AxisTracker.AxisFormat.NCF :
            permute_order = AxisTracker.AxisFormat.NFC_TO_NCF
            reverse_order = AxisTracker.AxisFormat.NCF_TO_NFC
        elif custom_axis_format == AxisTracker.AxisFormat.TNF and initial_axis_format== AxisTracker.AxisFormat.NTF :
            permute_order = AxisTracker.AxisFormat.TNF_TO_NTF
            reverse_order = AxisTracker.AxisFormat.NTF_TO_TNF
        elif custom_axis_format == AxisTracker.AxisFormat.NTF and initial_axis_format== AxisTracker.AxisFormat.TNF :
            permute_order = AxisTracker.AxisFormat.NTF_TO_TNF
            reverse_order = AxisTracker.AxisFormat.TNF_TO_NTF
        return permute_order, reverse_order

    def apply_custom_io_change(self, graph):
        for entry in graph.user_custom_io:
            buffer_name = str(entry['IOName'])
            log_assert(buffer_name in graph.buffers,"Incorrect IOName provided in custom IO YAML file. Buffer {} not found in graph"
                .format(buffer_name))
            isInput = False
            # Check if the buffer name provided is input buffer
            for node in graph.get_input_nodes_to_graph():
                if buffer_name in node.output_names:
                    isInput = True
                    break
            # Check if the buffer name provided is output buffer
            isOutput = False
            for node in graph.get_output_nodes_of_graph():
                if buffer_name in node.output_names:
                    isOutput = True
                    break
            log_assert((isInput or isOutput),"Custom IOName {} is neither an input nor an output.".format(buffer_name))

            if "Layout" in entry:
                entry['Layout']['Custom'] = InputLayout.get_axis_format(entry['Layout']['Custom'])
                initial_axis_format = graph.buffers[buffer_name].axis_format
                custom_axis_format = entry['Layout']['Custom']
                if initial_axis_format == custom_axis_format:
                    # No modification required in this case
                    continue

                if isInput:
                    self.modify_io_layout(graph, buffer_name, initial_axis_format, custom_axis_format, isInput = True)

                if isOutput:
                    self.modify_io_layout(graph, buffer_name, initial_axis_format, custom_axis_format, isInput = False)

            if "Datatype" in entry:
                if entry['Datatype'] in ['uint8','int8']:
                    # The code below is only to handle the conversions between non-quantized datatypes.
                    continue
                if isInput:
                    consumers = [str(name) for name in graph.buffers[buffer_name].consumers]
                    original_datatype = str(node.op.input_dtype)
                    if original_datatype == entry['Datatype']:
                        # No modification required in this case
                        continue

                    # In case of custom input datatype, replace the input node with a new node having the custom datatype
                    # and insert a cast op to cast the data to the datatype that the graph expects
                    new_op = op_adapter.InputOp(node.op.name, node.op.shape,
                                input_encoding_in = node.op.input_encoding_in,
                                input_encoding_out = node.op.input_encoding_out,
                                input_type = node.op.input_type,
                                input_dtype = entry['Datatype'])
                    graph.replace(node.op, new_op)
                    cast_op = op_adapter.CastOp(buffer_name+"_cast",to_type=original_datatype,from_type=entry['Datatype'])
                    graph.inject(cast_op, buffer_name, buffer_name+"_custom_cast", consumer_names=consumers)

                if isOutput:
                    # An output buffer of the original graph may no longer be an output buffer now due to the custom layout changes
                    # (permute node inserted) above. Hence, the code below sets the output buffer appropriately.
                    if graph.buffers[buffer_name].consumers:
                        new_output_node = list(graph.buffers[buffer_name].consumers)[0]
                        buffer_name = new_output_node.output_names[0]

                    # In case of custom output datatype, inject a cast op after the output and replace the original output
                    # with the output of cast op in graph.output_names list
                    cast_op = op_adapter.CastOp(buffer_name+"_cast",to_type=entry['Datatype'])
                    new_output_node = graph.inject(cast_op, buffer_name, buffer_name+"_custom_cast", consumer_names=[])
                    new_output_buffer_name = new_output_node.output_names[0]
                    graph.output_names.remove(buffer_name)
                    graph.output_names.append(new_output_buffer_name)

    def validate_custom_io_config(self, graph):
        buffers = set()
        for entry in graph.user_custom_io:
            for field in entry.keys():
                if field not in ['IOName', 'Layout', 'Datatype', 'QuantParam']:
                    log_error("Incorrect field %s provided in the custom IO YAML file. Valid fields are: 'IOName', 'Layout', 'Datatype', 'QuantParam'",
                        field)
            log_assert('IOName' in entry,"No IOName provided in custom IO YAML file. IOName is a mandatory field")
            buffer_name = entry['IOName']
            log_assert(buffer_name not in buffers,"Multiple entries provided for buffer {} in custom IO YAML file".format(buffer_name))
            if 'Layout' in entry:
                log_assert(('Custom' in entry['Layout'] and 'Model' in entry['Layout']),
                    "Both Custom layout and Model layout should be provided in the custom IO YAML file.")
            custom_datatype = None
            if 'Datatype' in entry:
                custom_datatype = entry['Datatype']
                log_assert((custom_datatype in ['float32','float16','int32','uint32','uint8','int8']),"Custom datatpe {} of buffer {} is not supported."
                    .format(entry['Datatype'], buffer_name))
            if 'QuantParam' in entry:
                quant_param = entry['QuantParam']
                for sub_field in quant_param.keys():
                    if sub_field not in ['Type', 'Scale', 'Offset']:
                        log_error("Incorrect field %s provided in QuantParam of the custom IO YAML file for buffer %s. Valid fields are: 'Type', 'Scale', 'Offset'",
                            field, buffer_name)
                log_assert('Type' in quant_param,"No Type provided for QuantParam in custom IO YAML file for buffer {}."
                    .format(buffer_name))
                log_assert(quant_param['Type'] == 'QNN_DEFINITION_DEFINED',"Type must be set to 'QNN_DEFINITION_DEFINED' if user wants to provide scale and offset.\
                    Invalid Type provided for buffer {}".format(buffer_name))
                log_assert('Scale' in quant_param and 'Offset' in quant_param,"Scale and/or Offset not provided for buffer {} in the custom IO YAML file."
                    .format(buffer_name))
                if custom_datatype is not None:
                    log_assert(custom_datatype in ['uint8', 'int8'], "Valid datatypes for quantized input/output are int8 and uint8 for custom IO.\
                        Invalid datatype {} provided for buffer {}.".format(custom_datatype, buffer_name))
            buffers.add(buffer_name)

class OptimizationTranslationBase(translation.Translation):
    """
    This class is to be used to perform graph optimizations such as: folding, squashing,pruning, etc. Additionally,
    it is also used to perform axis tracking and by default implements to spatial first order function
    (NCHW to NHWC, or TNF to NTF). Use this base class to get the default function and call register_method to add a new
    optimization. For eg: The OptimizeBatchnormTranslation overloads the axes_to_spatial_first_order to handle weights
    as well as adds a squash_batchnorm function and registers the method in the __init__ function.
    """
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(AXES_TO_SPATIAL_FIRST_ORDER, self.axes_to_spatial_first_order)
        self.register_method(MERGE_LOW_LEVEL_OPS_TO_LAYERS, self.merge_low_level_ops_to_layers)

    def axes_to_spatial_first_order(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        """
        Performs axis permutations(as needed) to get a spatial first order.

        Note: The eltwise_...() function that gets called re-populates the node's buffer "axis_format" and "shape" from
        source framework to the destination for certain ranks. If an overload of this function is done for a child class
        and this eltwise_...() function is not called make sure to understand and implement these changes to avoid
        conversion errors.

        :param node: an OpNode object to optimize from the IR graph
        :param graph: an IROpgraph object

        returns: True if any changes were done
                 False if no changes required
        """
        if AxisTracker.input_axis_formats_intact(graph, node, input_nontrivial_as_changed=True):
            # No change in input formats, and none of the input formats are NonTrivial
            # Nothing to do in this case
            return False

        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        return True

    def merge_low_level_ops_to_layers(self, graph):
        """"
        When overloaded in the child class, it is implemented to merge to the low level ops to layers.

        """
        pass


# ------------------------------------------------------------------------------------------------------------------
#   Graph Optimizations
# ------------------------------------------------------------------------------------------------------------------
def register_graph_optimization(graph_optimization_method):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param graph_optimization_method: a concrete class for a given optimization
    """
    return graph_optimization_method


@register_graph_optimization
def remove_disconnected_nodes(graph):
    """Removes nodes with all its outputs unconsumed from the graph."""
    all_ops = set(graph.nodes_in_order)
    connected_ops = set()
    queue = []
    graph_output_nodes = graph.get_output_nodes_of_graph()

    if graph_output_nodes:
        queue.extend(graph_output_nodes)
        # Find nodes from Output to Input Op
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node and filter out null input
            node_inputs = [node_input for node_input in graph.get_op_input_nodes(node) if node_input]
            new_nodes = [node_ for node_ in node_inputs if (node_ not in connected_ops and node_ not in queue)]
            queue.extend(new_nodes)

    else:
        # Ensure input nodes have consumers before adding them to queue
        input_nodes = graph.get_input_nodes_to_graph()
        input_nodes = [node for node in input_nodes if graph.get_buffer(node.output_names[0]).consumers]
        queue.extend(input_nodes)
        # Find nodes from Input Op to outputs
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node, this will add the Constant input Ops that will be otherwise missed
            node_inputs = [node_input for node_input in graph.get_op_input_nodes(node) if node_input]
            new_nodes = [node for node in node_inputs if node not in connected_ops]
            for new_node in new_nodes:
                queue.insert(0, new_node)

            # Extend the queue with output nodes
            node_outputs = graph.get_op_output_nodes(node)
            new_nodes = [node for node in node_outputs if node not in queue]
            queue.extend(new_nodes)

    disconnected_nodes = all_ops - connected_ops
    prunable_node_names = [node.op.name for node in disconnected_nodes]
    if disconnected_nodes:
        log_debug("Pruning Disconnected nodes {}".format(prunable_node_names))

    for node in disconnected_nodes:
        try:
            graph.prune(node, force_remove=True)
        except Exception as e:
            log_error("Cannot find node {}".format(node.op.name))
            raise e

    if not graph.list_nodes():
        raise ValueError("After pruning disconnected nodes, this model is empty.")

    return graph


# ------------------------------
# Util used for common squashing
# ------------------------------
def squash_node_into_nn_node(graph, matched_node_list):
    """
    Squashes a node into an NN node. This can be done by accounting for the node's operation in arithmetic adjustments
    to the NN node's weights and biases. Intended use is for Elementwise ops that follow an NN op.
    :param graph: The IROpGraph object
    :param matched_node_list: the list of nodes that contain elementwise ops, have a constant input, and are
                              preceded by a node that contains an NN op
    """

    OPS_HAVING_BIAS_SUM = [
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]
    ]
    OPS_HAVING_BIAS_SUB = [
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT]
    ]
    OPS_HAVING_WEIGHTS_PRODUCT = [
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY]
    ]
    OPS_HAVING_WEIGHTS_DIV = [
        op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE]
    ]

    for node_tuple in matched_node_list:
        # collect previous and current op information
        node = node_tuple[0]
        node_type = node.op.type
        nn_buf, nn_op, const_op = None, None, None
        for name in node.input_names:
            input_buf = graph.get_buffer(name)
            input_op = graph.get_producer_op(name)
            if (len(input_buf.producer.input_names) == 3 and
                input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                  op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                  op_adapter.FullyConnectedOp.TRANSLATION_KEY]) or \
                    (hasattr(input_op, "weights") or hasattr(input_op, "bias")):
                # temp fix to avoid squashing of eltwise Ops into Matmul
                # TODO: Remove once Matmul Opdef is updated to support bias attribute
                if input_op.type == op_adapter.MatMulOp.TRANSLATION_KEY:
                    return
                nn_buf = input_buf
                nn_op = input_op
                if len(nn_buf.producer.input_names) == 3 and \
                        nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                       op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                       op_adapter.FullyConnectedOp.TRANSLATION_KEY]:
                    manage_shared_static_input(graph, nn_buf.producer, 1)
                    src_weight = graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.tensor
                    manage_shared_static_input(graph, nn_buf.producer, 2)
                    src_bias = graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor
                else:
                    src_weight = nn_op.weights
                    src_bias = nn_op.bias
            elif input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                const_op = input_op

        if nn_op is None:
            raise ValueError("Failed to retrieve NN op to squash {} node {} into.".format(node_type, node.op.name))

        if const_op is None:
            raise ValueError("Failed to retrieve const op to squash {} node {} into.".format(node_type, node.op.name))

        if nn_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            if (len(nn_buf.producer.input_names) == 4 or nn_op.hasattr("weights")) and len(src_weight.shape) == 5:
                # weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for broadcasting to handle non-square kernel and then revert
                if nn_op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv3d_weights_to_ir)
                elif nn_op.type == op_adapter.TranposeConv2dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv3d_weights_to_ir)
            if const_op is not None and len(const_op.tensor.shape) == 5:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
        elif nn_buf.axis_format == AxisTracker.AxisFormat.NCS:
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("weights")) and len(src_weight.shape) == 4:
                # weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for broadcasting to handle non-square kernel and then revert
                if nn_op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv2d_weights_to_ir)
                elif nn_op.type == op_adapter.TransposeConv2dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv2d_weights_to_ir)
            if const_op is not None and len(const_op.tensor.shape) == 4:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC)

        # separate conditionals according to which arithmetic operation needs to happen
        if node_type in OPS_HAVING_BIAS_SUM:
            scale_bias = const_op.tensor
            src_bias = np.atleast_1d((src_bias + scale_bias).squeeze())
        elif node_type in OPS_HAVING_BIAS_SUB:
            scale_bias = const_op.tensor
            src_bias = np.atleast_1d((src_bias - scale_bias).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_PRODUCT:
            scale_weights = const_op.tensor
            src_weight = src_weight * scale_weights
            src_bias = np.atleast_1d((src_bias * scale_weights).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_DIV:
            scale_weights = const_op.tensor
            src_weight = src_weight / scale_weights
            src_bias = np.atleast_1d((src_bias / scale_weights).squeeze())
        else:
            raise ValueError("Squashing {} node {} into {} node {} unsupported.".format(node_type, node.op.name,
                                                                                        nn_op.type, nn_op.name))

        if nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY]:
            src_weight= np.atleast_1d(src_weight.squeeze())

        if nn_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            if (len(input_buf.producer.input_names) == 4 or nn_op.hasattr("weights")) and len(src_weight.shape) == 5:
                if nn_op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv3d_weights_from_ir)
                elif nn_op.type == op_adapter.TransposeConv2dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv3d_weights_from_ir)
            if const_op is not None and len(const_op.tensor.shape) == 5:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NDHWC_TO_NCDHW)
        elif nn_buf.axis_format == AxisTracker.AxisFormat.NCS:
            if (len(input_buf.producer.input_names) == 3 or nn_op.hasattr("weights")) and len(src_weight.shape) == 4:
                if nn_op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv2d_weights_from_ir)
                elif nn_op.type == op_adapter.TransposeConv2dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv2d_weights_from_ir)
            if const_op is not None and len(const_op.tensor.shape) == 4:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NSC_TO_NCS)

        if len(nn_buf.producer.input_names) == 3 and \
                nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                               op_adapter.InstanceNormOp.TRANSLATION_KEY,
                               op_adapter.FullyConnectedOp.TRANSLATION_KEY]:
            origin_bias_tensor = graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor
            graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.tensor = src_weight
            graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor = src_bias
        else:
            origin_bias_tensor = nn_op.bias
            nn_op.weights = src_weight
            nn_op.bias = src_bias
        log_debug2(code_to_message.get_debugging_message("DEBUG_SQUASH_INTO_NN_NODE")
                   (node_type, node.op.name, nn_op.type, nn_op.name))

        # for conv_node without bias, graph will prepare bias for it (zero bias),
        # however its true bias name is stored in eltwise node. for better user experience,
        # we should use eltwise bias name instead of prepared one
        if np.allclose(origin_bias_tensor, 0) and node_type in OPS_HAVING_BIAS_SUM:
            bias_buffer = graph.get_buffer(nn_buf.producer.input_names[2])
            eltwise_bias_name = const_op.name

            # remove nn_node from nn_node.bias's consumers
            bias_buffer.consumers.remove(nn_buf.producer)
            if len(bias_buffer.consumers)==0:
                graph.prune(bias_buffer.producer)

            # change nn_node input_names[2] to eltwise_node.bias
            nn_buf.producer.input_names[2] = eltwise_bias_name
            eltwise_bias_buffer = graph.get_buffer(eltwise_bias_name)
            eltwise_bias_buffer.consumers.add(nn_buf.producer)
            eltwise_bias_buffer.axis_format = bias_buffer.axis_format

            # weight, nn_node, bias => weight, bias, nn_node
            idx_nn = graph.nodes_in_order.index(nn_buf.producer)
            idx_bias = graph.nodes_in_order.index(eltwise_bias_buffer.producer)
            if idx_nn < idx_bias:
                graph.nodes_in_order[idx_nn] = eltwise_bias_buffer.producer
                graph.nodes_in_order[idx_bias] = nn_buf.producer
        graph.squash(node, input_name=nn_buf.name)


def validate_eltwise_pattern(graph, nodes_tuple, mode):
    """
    Common function to validate if pattern is squashable
    :param graph: the IROpGraph
    :param nodes_tuple: the matched list of nodes
    :param mode: either bias or weight. Use to determine if squashing is
                 eltwise[add|sub] or eltwise[prod|div] respectively.
    :return:
    """

    OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS = [
        op_adapter.Conv2dOp.TRANSLATION_KEY,
        op_adapter.TransposeConv2dOp.TRANSLATION_KEY,
        op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY
    ]

    node = nodes_tuple[0]
    nn_buf, nn_op, const_op = None, None, None
    is_batchnorm_input = False
    is_fully_connected_input = False
    for name in node.input_names:
        input_node = graph.get_buffer(name).producer
        input_op = graph.get_buffer(name).producer.op
        # verify that one of the inputs is constant and the other input is produced by nn_type op(BN, FC, Conv/Deconv)
        if input_op.type in OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS:
            # Squashing elementwise operations into these nodes is handled in their respective optimizations classes
            return False
        elif input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op
        elif (mode == "weights" and hasattr(input_op, "weights") and hasattr(input_op, "bias")) or \
                (mode == "bias" and hasattr(input_op, "bias")) or \
                    (len(input_node.input_names) == 3 and
                     input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY, op_adapter.FullyConnectedOp.TRANSLATION_KEY]):
            if len(graph.get_buffer(name).consumers) != 1:
                # Unable to squash into nn_op which has more than one consumer
                return False
            nn_op = input_op
            nn_buf = graph.get_buffer(name)
            if input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY]:
                is_batchnorm_input = True
            if input_op.type == op_adapter.FullyConnectedOp.TRANSLATION_KEY:
                is_fully_connected_input = True

    # For mode:weights
    #      Only valid to squash if the nn_op has act output that are broadcastable with the scale weights AND
    #      the scale weights are same rank with nn_op bias and broadcastable
    # For mode:bias
    #      Only valid if the nn_op has a bias with the same rank as const_op and broadcastable
    if nn_op is not None and const_op is not None:
        const_shape = const_op.tensor.shape
        const_shape_squeezed = np.atleast_1d(const_op.tensor.squeeze()).shape
        if is_batchnorm_input:
            bias_shape = graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.tensor.shape
        elif is_fully_connected_input:
            bias_shape = graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor.shape
        else:
            bias_shape = nn_op.bias.shape
        if mode == 'bias':
            if len(const_shape_squeezed) == len(bias_shape) and \
                    translation_utils.broadcastable(bias_shape, const_shape_squeezed):
                return True
        elif mode == 'weights':
            nn_buf_shape = nn_buf.get_buf_dims()
            axis_order = graph.src_axis_order
            input_ir_shapes = [axis_order.permute_shape_to_ir(nn_buf_shape),
                               axis_order.permute_shape_to_ir(const_shape)]
            # Note: verify with the ir shapes for inputs since this is done pre axis-tracking
            if translation_utils.broadcastable(*input_ir_shapes) and \
                    (len(const_shape_squeezed) == len(bias_shape) and
                     translation_utils.broadcastable(bias_shape, const_shape_squeezed)):
                return True
    return False


def add_or_broadcast_bias(node, graph, output_channel):
    weights_buffer = graph.get_buffer(node.input_names[1])
    if len(node.input_names) < 3:
        bias_tensor = np.zeros([output_channel], dtype=np.float32)
        bias_op_name = node.op.name + "_bias"
        bias_op = op_adapter.ConstantOp(bias_op_name, tensor=bias_tensor.copy())
        conv_idx = graph.list_nodes().index(node)
        graph.add(bias_op, [], [bias_op_name], axis_formats=[AxisTracker.AxisFormat.ANY], idx=conv_idx)
        graph.get_buffer(bias_op_name).consumers.add(node)
        node.input_names.append(bias_op_name)
    else:
        bias_buffer = graph.get_buffer(node.input_names[2])
        # Represents case where broadcasting biases is required
        if bias_buffer.shape[0] < output_channel:
            bias_const_node = bias_buffer.producer
            if len(bias_const_node.op.tensor) != 1:
                raise ValueError("Unable to broadcast bias tensor for node {}".format(node.op.name))
            bias_const_node.op.tensor = np.repeat(bias_const_node.op.tensor, weights_buffer.shape[3])
            bias_buffer.shape = list(bias_const_node.op.tensor.shape)


def squash_eltwise_into_conv(graph, conv_node):
    conv_output_buffer = graph.get_buffer(conv_node.output_names[0])
    eltwise_node = list(conv_output_buffer.consumers)[0]
    # Find and assign the const_op from eltwise_node's input_names
    const_op = None
    for name in eltwise_node.input_names:
        input_op = graph.get_producer_op(name)
        if input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op

    # Ensure the constant operation has the proper squash shape based on source axis order
    const_tensor = const_op.tensor
    if conv_output_buffer.axis_format == AxisTracker.AxisFormat.NCDHW and len(const_op.tensor.shape) == 5:
        const_tensor = np.transpose(const_tensor, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
    elif conv_output_buffer.axis_format == AxisTracker.AxisFormat.NCS and len(const_op.tensor.shape) == 4:
        const_tensor = np.transpose(const_tensor, AxisTracker.AxisFormat.NCS_TO_NSC)

    manage_shared_static_input(graph, conv_node, 2)
    bias_buffer = graph.get_buffer(conv_node.input_names[2])
    bias_producer = bias_buffer.producer
    bias_tensor = bias_producer.op.tensor
    origin_bias_tensor = bias_tensor

    # Apply the const_node's tensor to the conv_node's bias according to type of elementwise operation
    if eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]:
        bias_tensor = np.atleast_1d((bias_tensor + const_tensor).squeeze())
    elif eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT]:
        bias_tensor = np.atleast_1d((bias_tensor - const_tensor).squeeze())
    else:
        # Only ElementwiseProduct/DivOp require static weights, so extract the static weights only in these cases
        manage_shared_static_input(graph, conv_node, 1)
        weights_producer = graph.get_buffer(conv_node.input_names[1]).producer
        weights_tensor = weights_producer.op.tensor
        if eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY]:
            weights_tensor = weights_tensor * const_tensor
            bias_tensor = np.atleast_1d((bias_tensor * const_tensor).squeeze())
        elif eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE]:
            weights_tensor = weights_tensor / const_tensor
            bias_tensor = np.atleast_1d((bias_tensor / const_tensor).squeeze())
        weights_producer.op.tensor = weights_tensor

    # Reincorporate the new bias and squash the elementwise operation
    bias_producer.op.tensor = bias_tensor
    log_debug2(code_to_message.get_debugging_message("DEBUG_SQUASH_INTO_NN_NODE")
               (eltwise_node, eltwise_node.op.name, conv_node.op.type, conv_node.op.name))

    # for conv_node without bias, graph will prepare bias for it (zero bias),
    # however its true bias name is stored in eltwise node. for better user experience,
    # we should use eltwise bias name instead of prepared one
    if np.allclose(origin_bias_tensor, 0) and \
        eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]:
        eltwise_bias_name = eltwise_node.input_names[1]

        # change conv_node input_names[2] to eltwise_node.bias
        conv_node.input_names[2] = eltwise_bias_name
        eltwise_bias_buffer = graph.get_buffer(eltwise_bias_name)
        eltwise_bias_buffer.consumers.add(conv_node)

        # remove conv_node from conv.bias's consumers
        bias_buffer.consumers.remove(conv_node)
        if len(bias_buffer.consumers)==0:
            graph.prune(bias_producer)

        # reshape bias tensor to 1d
        eltwise_bias_node = eltwise_bias_buffer.producer
        eltwise_bias_node.op.tensor = np.atleast_1d(eltwise_bias_node.op.tensor.squeeze())
        eltwise_bias_buffer.shape = list(eltwise_bias_node.op.tensor.shape)
        eltwise_bias_buffer.axis_format = bias_buffer.axis_format

        # weight, conv, bias => weight, bias, conv
        idx_conv = graph.nodes_in_order.index(conv_node)
        idx_bias = graph.nodes_in_order.index(eltwise_bias_node)
        if idx_conv < idx_bias:
            graph.nodes_in_order[idx_conv] = eltwise_bias_node
            graph.nodes_in_order[idx_bias] = conv_node
    graph.squash(eltwise_node, input_name=conv_output_buffer.name)


def validate_conv_eltwise_pattern(graph, conv_node, eltwise_type):
    conv_node_output_buffer = graph.get_buffer(conv_node.output_names[0])
    if len(conv_node_output_buffer.consumers) != 1 or \
            list(conv_node_output_buffer.consumers)[0].op.type != eltwise_type:
        return False

    eltwise_node = list(conv_node_output_buffer.consumers)[0]

    # Find the constant op from input_names of the eltwise_node
    const_op = None
    for name in eltwise_node.input_names:
        input_op = graph.get_producer_op(name)
        if input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op

    # Constant op was not found, so we cannot squash this elementwise operation
    if const_op is None:
        return False

    # Scalar products are able to be squashed into convolution weights
    if eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY]:
        return len(const_op.tensor.shape) == 1

    const_shape_squeezed = np.atleast_1d(const_op.tensor.squeeze()).shape
    bias_shape = graph.get_buffer(conv_node.input_names[2]).shape
    # Const shape and bias shape should have the same rank and be broadcastable
    return len(const_shape_squeezed) == len(bias_shape) and \
        translation_utils.broadcastable(bias_shape, const_shape_squeezed)


def prepare_conv_inputs_as_params(graph, conv_node):
    weights_buffer = graph.get_buffer(conv_node.input_names[1])
    weights_node = weights_buffer.producer
    bias_buffer = graph.get_buffer(conv_node.input_names[2])
    bias_node = bias_buffer.producer
    if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
            bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
        conv_node.op.weights = weights_node.op.tensor
        conv_node.op.bias = bias_node.op.tensor
        # Remove the weights/bias inputs from the IR graph
        graph.remove_node_as_consumer(conv_node, weights_buffer.name)
        graph.remove_node_as_consumer(conv_node, bias_buffer.name)
        conv_node.input_names = [conv_node.input_names[0]]


def manage_shared_static_input(graph, node, idx):
    """
    Create a copy of node's input[idx] if input[idx] has more than one consumer and
    assigns the copy as node's input[idx].
    """
    if not len(graph.get_buffer(node.input_names[idx]).consumers) > 1 :
        return
    input_buffer = graph.get_buffer(node.input_names[idx])
    producer_op = input_buffer.producer.op
    weight_tensor = producer_op.tensor
    weight_tensor_copy = np.copy(weight_tensor)
    if idx == 1:
        name = node.op.name + "_kernel_weight"
    else:
        name = node.op.name + "_kernel_bias"
    const_op = op_adapter.ConstantOp(name, tensor=weight_tensor_copy)
    producer_idx = graph.list_nodes().index(input_buffer.producer)
    graph.add(const_op, [], [name], axis_formats=[input_buffer.axis_format], idx=producer_idx+1)
    graph.get_buffer(name).consumers.add(node)
    input_buffer.consumers.remove(node)
    node.input_names[idx] = name
    # copy quant overrides to new const buffer only if it exists in original static input
    if(graph.has_quantization_param(input_buffer.name)):
        quant_param = graph.get_layer_quantization_param(input_buffer.name)
        graph.add_quantization_params(name,
                                      bn_params=quant_param['bn_params'],
                                      output_encodings=quant_param['output_encodings'],
                                      param_encodings=quant_param['param_encodings'])


# ------------------------------------------------------------------------------------------------------------------
#   Translations
#   Note: each Optimization Concrete class has at a minimum 1 optimize function. i.e axes_to_spatial_first_order(..)
#         if more is needed for a given op, it needs to register that method_key and implement a function for it.
# ------------------------------------------------------------------------------------------------------------------
def register_layer_optimization(layer_translation):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param layer_translation: a concrete class for a given optimization
    """
    OptimizationTranslations.register_translation(layer_translation(), layer_translation().op_type)
    return layer_translation


@register_layer_optimization
class OptimizeInputTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InputOp.TRANSLATION_KEY
        self.register_method(EXTRACT_COLOR_TRANSFROM, self.extract_color_transform)

    @staticmethod
    def extract_color_transform(graph):
        """ Optional Optimization to create separate Op to handle color transformation pre-processing for network
            inputs
        """
        def validate_transformation(nodes_tuple):
            node_ = nodes_tuple[0]
            if node_.op.input_encoding_in != node_.op.input_encoding_out and \
                    node_.op.input_encoding_in not in [InputEncodings.TIME_SERIES, InputEncodings.OTHER]:
                return True
            return False

        sequence = [("input", (), ())]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_transformation)

        for node_tuple in matched_node_list:
            input_node = node_tuple[0]
            # adjust shape for input as that will be the expected shape after transformation
            color_transform_name = input_node.output_names[0] + "_post_transform"
            color_transform_output_shape = input_node.op.shape

            input_buf = graph.get_buffer(input_node.output_names[0])
            old_input_format = input_buf.axis_format
            b, h, w, c = graph.src_axis_order.extract_2d_spatial_dims(input_node.op.shape)
            if input_node.op.input_encoding_in in (InputEncodings.NV21, InputEncodings.NV12):
                # determine expected shape for yuv_(nv21|nv12)(width * height * 3 / 2)
                shape = int(h * w * (3 / 2))
                input_node.op.shape = [input_node.op.shape[0], shape]
                input_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                b, h, w, c = graph.src_axis_order.extract_2d_spatial_dims(input_node.op.shape)
                input_node.op.shape = graph.src_axis_order.format_2d_spatial_output_shape(b, h, w, 4)
            input_buf.set_buf_dims(input_node.op.shape)

            color_transform_op = op_adapter.ColorTransformOp(color_transform_name,
                                                             color_transform_output_shape,
                                                             input_encoding_in=input_node.op.input_encoding_in,
                                                             input_encoding_out=input_node.op.input_encoding_out)
            graph.inject(color_transform_op,
                         input_name=input_node.output_names[0],
                         output_name=color_transform_name,
                         axis_format=old_input_format)
            log_debug2(code_to_message.get_debugging_message("DEBUG_COLOR_TRANSFORM_EXTRACTION")
                       (input_node.op.name, input_node.op.shape, input_node.op.input_encoding_in))

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
            buf.axis_format = AxisTracker.AxisFormat.NDHWC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.NCF:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCF_TO_NFC)
            buf.axis_format = AxisTracker.AxisFormat.NFC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.TNF:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            buf.axis_format = AxisTracker.AxisFormat.NTF
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.OIDHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.OIDHW_TO_DHWIO)
            buf.axis_format = AxisTracker.AxisFormat.DHWIO
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.OIHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.OIHW_TO_HWIO)
            buf.axis_format = AxisTracker.AxisFormat.HWIO
            node.op.shape = buf.shape
        return True


class Optimize1DNNTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.nn_2d_op = None
        self.idx_to_insert = 0

    @staticmethod
    def add_reshape_op(graph, reshape_op_name, output_shape, input_names, output_names, axis_formats=None, idx=-1):
        node = graph.add(op_adapter.ReshapeOp(reshape_op_name, shape=output_shape), input_names, output_names,
                         axis_formats=axis_formats, idx=idx)
        input_buffers = graph.get_input_buffers(node)
        node.op.data_axis_formats = [in_buf.axis_format for in_buf in input_buffers]

    def setup_for_1d_to_2d_nn_replacement(self, node, graph):
        # compute the correct idx to insert new nodes
        self.idx_to_insert = 0
        for input_name in node.input_names:
            buf = graph.get_buffer(input_name)
            cur_idx = graph.nodes_in_order.index(buf.producer)
            if self.idx_to_insert < cur_idx:
                self.idx_to_insert = cur_idx
        self.idx_to_insert = self.idx_to_insert + 1

    def reshape_to_2d_spatial(self, input_name, node, graph):
        reshape_2d_op_name = node.op.name + '_reshape_to_2d'
        buffer = graph.get_buffer(input_name)
        output_shape = buffer.shape + [1]
        # add Reshape to transform NN Op input from 1D to 2D spatial dimension
        self.add_reshape_op(graph, reshape_2d_op_name, output_shape, [input_name],
                            [reshape_2d_op_name], axis_formats=[AxisTracker.AxisFormat.NCS], idx=self.idx_to_insert)

        # increment idx_to_insert
        self.idx_to_insert = self.idx_to_insert + 1
        return reshape_2d_op_name

    def reshape_weights(self, input_name, node, graph):
        weight_buffer = graph.get_buffer(input_name)
        if len(weight_buffer.shape) == 3:
            output_shape = weight_buffer.shape + [1]
            if weight_buffer.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                weight_buffer.producer.op.tensor = weight_buffer.producer.op.tensor.reshape(output_shape)
                weight_buffer.shape = output_shape
                if node.op.type == op_adapter.TransposeConv1dOp.TRANSLATION_KEY:
                    weight_buffer.axis_format = graph.src_axis_order.deconv2d_weights_format
                else:
                    weight_buffer.axis_format = graph.src_axis_order.conv2d_weights_format
                return input_name
            else:
                # add Reshape to transform weight input from 1D to 2D spatial dimension
                weights_reshape_op_name = node.op.name + '_reshaped_weights'
                if node.op.type == op_adapter.TransposeConv1dOp.TRANSLATION_KEY:
                    axis_formats = [AxisTracker.AxisFormat.IOHW]
                else:
                    axis_formats = [AxisTracker.AxisFormat.OIHW]
                self.add_reshape_op(graph, weights_reshape_op_name, output_shape,
                                    [input_name], [weights_reshape_op_name], axis_formats=axis_formats,
                                    idx=self.idx_to_insert)
                # increment idx_to_insert
                self.idx_to_insert = self.idx_to_insert + 1
                return weights_reshape_op_name
        # no reshape needed return original input name
        return input_name

    def add_nn_2d_node(self, input_names, node, graph):
        nn_2d_output_name = node.op.name + '_intermediate'
        # add 2D NN Op to the graph
        graph.add(self.nn_2d_op, input_names, [nn_2d_output_name], idx=self.idx_to_insert)
        # increment idx_to_insert
        self.idx_to_insert = self.idx_to_insert + 1
        return nn_2d_output_name

    def reshape_to_1d_spatial(self, nn_2d_output_name, output_names, nn1d_consumers, graph, pos=None):
        nn_2d_output_buffer = graph.get_buffer(nn_2d_output_name)
        output_shape = nn_2d_output_buffer.shape[:-1]
        # add Reshape to transform NN Op output from 2D back to 1D spatial dimension
        self.add_reshape_op(graph, nn_2d_output_name, output_shape, [nn_2d_output_name],
                            [output_names[0]], axis_formats=[AxisTracker.AxisFormat.NCF], idx=self.idx_to_insert)

        # add back 1D NN node's consumers to reshape's output buffer
        output_buffer = graph.get_buffer(output_names[0])
        for i, c in enumerate(nn1d_consumers):
            if pos == None:
                c.input_names.insert(0, output_names[0])
            else:
                c.input_names.insert(pos[i], output_names[0])
            output_buffer.consumers.add(c)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        pos = []
        nn_2d_input_names = node.input_names[:]
        nn1d_consumers = graph.get_buffer(node.output_names[0]).consumers

        self.setup_for_1d_to_2d_nn_replacement(node, graph)

        # reshape nn_1d inputs and update input_names for nn_2d op
        nn_2d_input_names[0] = self.reshape_to_2d_spatial(node.input_names[0], node, graph)
        if len(node.input_names) > 1:
            # reshape weights if applicable
            nn_2d_input_names[1] = self.reshape_weights(node.input_names[1], node, graph)

        # add 2d variant for nn_op as intermediate op
        nn_2d_output_name = self.add_nn_2d_node(nn_2d_input_names, node, graph)
        for _ , c in enumerate(nn1d_consumers):
            pos.append(c.input_names.index(node.output_names[0]))
        # prune the 1D NN node
        graph.prune(node, force_remove=True)

        # post reshape to mimic nn_1d output shape
        self.reshape_to_1d_spatial(nn_2d_output_name, node.output_names, nn1d_consumers, graph, pos)


@register_layer_optimization
class OptimizeArgOpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            axis_map = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Optimize special case that channel dimension is removed
                if node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                node.op.axis = axis_map[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            axis_map = AxisTracker.AxisFormat.NSC_TO_NCS
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Optimize special case that channel dimension is removed
                if node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                node.op.axis = axis_map[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            axis_map = AxisTracker.AxisFormat.NFC_TO_NCF
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Channel dimension is removed
                if node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NF
                # Feature dimension is removed
                elif node.op.axis == 2:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NC
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                                  AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                node.op.axis = axis_map[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            axis_map = AxisTracker.AxisFormat.NTF_TO_TNF
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Time dimension is removed
                if node.op.axis == 0:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NF
                # Batch dimension is removed
                elif node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                                  AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                node.op.axis = axis_map[node.op.axis]
        else:
            # Add warning message for other axis formats
            log_warning("No need to handle other axis formats now, but might optimize them in the future.")
        return True


@register_layer_optimization
class OptimizeBatchnormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchnormOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if 1 < input_buf.rank() <= 5:
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            output_buffer = graph.get_output_buffers(node)[0]
            # (image/feature)_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
            # Enforce the output format here to be NDHWC/NSC/NFC
            output_buffer.axis_format = AxisOrder().get_axis_format(len(output_buffer.shape))
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_BATCHNORM_DIM_UNSUPPORTED")(input_buf.rank(),
                                                                                                  node.op.name))
        return True

    def merge_low_level_ops_to_layers(self, graph):
        def validate(nodes_tuple_):
            prod_node = nodes_tuple_[1]
            prod_node_input_op = graph.get_producer_op(prod_node.input_names[0])
            # previous must not be a Batchnorm and previous node must be a nn_node for sequence to match batchnorm
            if prod_node_input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY] or \
                    (prod_node_input_op.type != op_adapter.FullyConnectedOp.TRANSLATION_KEY and \
                        (not hasattr(prod_node_input_op, "weights") or not hasattr(prod_node_input_op, "bias"))):
                return False

            mul_const_ip_node_ = nodes_tuple_[0]
            add_const_ip_node_ = nodes_tuple_[2]
            # batchnorm nodes require 1D weights/biases
            mul_const_ip_ = np.atleast_1d(mul_const_ip_node_.op.tensor.squeeze())
            add_const_ip_ = np.atleast_1d(add_const_ip_node_.op.tensor.squeeze())
            if len(mul_const_ip_.shape) != 1 or len(add_const_ip_.shape) != 1:
                return False
            return True

        sequence = [
                    ("constant", (), ()),
                    ("elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
                        ()),
                    ("constant", (), ()),
                    ("elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0),
                                                 ("constant", 1)]), ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            mul_const_ip_node = nodes_tuple[0]
            mul_node = nodes_tuple[1]
            add_const_ip_node = nodes_tuple[2]
            add_node = nodes_tuple[3]

            # batchnorm nodes require 1D weights/biases
            mul_const_ip = np.atleast_1d(mul_const_ip_node.op.tensor.squeeze())
            add_const_ip = np.atleast_1d(add_const_ip_node.op.tensor.squeeze())

            # Squashes the add node
            add_input_buffer = graph.get_input_buffers(add_node)[0]
            graph.squash(add_node, input_name=add_input_buffer.name)

            # Remove mul_node as consumer of const node's buffer
            graph.get_buffer(mul_const_ip_node.output_names[0]).consumers.remove(mul_node)
            # Remove const node from mul_node's input names
            mul_node.input_names.remove(mul_const_ip_node.output_names[0])
            # Change the weight/bias to a constant node
            weights_name = mul_node.op.name + "_bn_w"
            bias_name = mul_node.op.name + "_bn_b"
            weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=mul_const_ip)
            bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=add_const_ip)
            graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            # Replace the mul node to an batchnorm node

            batchnorm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.BatchnormOp.type,
                                                                        op_adapter.BatchnormOp.LEGACY_TRANSLATION_KEY)
            batchnorm_op = op_adapter.BatchnormOp(batchnorm_op_name)

            graph.replace(mul_node.op, batchnorm_op)
            batchnorm_node = graph.nodes_by_name[batchnorm_op.name]
            batchnorm_node.input_names.append(weights_name)
            batchnorm_node.input_names.append(bias_name)
            graph.get_buffer(weights_name).consumers.add(batchnorm_node)
            graph.get_buffer(bias_name).consumers.add(batchnorm_node)

    @staticmethod
    def squash_batchnorm(graph):
        def validate(nodes_tuple):
            bn_node_ = next(iter(graph.get_output_buffers(nodes_tuple[0])[0].consumers))
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            return bn_node_.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY and bn_input_buffer_.rank() >= 4

        sequences = [[("Conv2d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))],
                     [("DepthWiseConv2d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))],
                     [("TransposeConv2d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))]
                    ]

        for sequence in sequences:
            for node_tuple in graph.get_matched_nodes(sequence, validator=validate):
                # sanity check
                log_assert(len(node_tuple) == len(sequence),
                           "Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                           len(node_tuple), len(sequence))

                conv_node = node_tuple[0]
                bn_node = next(iter(graph.get_output_buffers(conv_node)[0].consumers))
                bn_input_buffer = graph.get_input_buffers(bn_node)[0]
                bn_node_weights = graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
                bn_node_bias = graph.get_buffer(bn_node.input_names[2]).producer.op.tensor

                manage_shared_static_input(graph, conv_node, 1)
                conv_node_weights_buffer = graph.get_buffer(conv_node.input_names[1])
                conv_node_weights_op = conv_node_weights_buffer.producer.op
                conv_node_weights = conv_node_weights_op.tensor

                # Extract bias from ConstantOp
                manage_shared_static_input(graph, conv_node, 2)
                conv_node_bias_op = graph.get_buffer(conv_node.input_names[2]).producer.op
                conv_node_bias = conv_node_bias_op.tensor

                if conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.OIDHW:
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_conv3d_weights_to_ir)
                    weights = weights * bn_node.op.weights
                    weights = np.transpose(weights, graph.src_axis_order.permute_conv3d_weights_from_ir)
                elif conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.OIHW:
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_conv2d_weights_to_ir)
                    weights = weights * bn_node_weights
                    weights = np.transpose(weights, graph.src_axis_order.permute_conv2d_weights_from_ir)
                elif conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.IOHW:
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_deconv2d_weights_to_ir)
                    weights = weights * bn_node_weights
                    weights = np.transpose(weights, graph.src_axis_order.permute_deconv2d_weights_from_ir)
                else:
                    weights = conv_node_weights * bn_node_weights

                conv_node_weights_op.tensor = weights
                conv_node_bias = np.atleast_1d(
                    (conv_node_bias * bn_node_weights + bn_node_bias).squeeze())
                conv_node_bias_op.tensor = conv_node_bias.copy()

                # add cached bn parameters as conv node supplemental attributes before squashing
                if graph.has_quantization_param(bn_node.op.name) and \
                        (conv_node.op.type == op_adapter.Conv2dOp.TRANSLATION_KEY or \
                            conv_node.op.type == op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY):
                    gamma_data = graph.quantization_params[bn_node.op.name]["bn_params"]["gamma"]
                    beta_data = graph.quantization_params[bn_node.op.name]["bn_params"]["beta"]
                    gamma = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_PARAM_BN_GAMMA,
                                                    gamma_data.shape,
                                                    gamma_data,
                                                    ir_graph.QNN_DATATYPE_FLOAT_32)
                    beta = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_PARAM_BN_BETA,
                                                   beta_data.shape,
                                                   beta_data,
                                                   ir_graph.QNN_DATATYPE_FLOAT_32)
                    attrs = conv_node.op.c_op.attrs
                    attrs.add(ir_graph.IR_OP_CONV_PARAM_BN_GAMMA, gamma, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                    attrs.add(ir_graph.IR_OP_CONV_PARAM_BN_BETA, beta, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

                graph.squash(bn_node, input_name=bn_input_buffer.name)
                log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                           conv_node.op.type,
                                                                                           conv_node.op.name))

    def prepare_inputs_as_params(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        bias_buffer = graph.get_buffer(node.input_names[2])
        bias_node = bias_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
            node.op.bias = bias_node.op.tensor
            # Remove the weights/bias inputs from the IR graph
            graph.remove_node_as_consumer(node, weights_buffer.name)
            graph.remove_node_as_consumer(node, bias_buffer.name)
            node.input_names = [node.input_names[0]]


@register_layer_optimization
class OptimizeBatchPermutationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchPermutationOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCastTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CastOp.TRANSLATION_KEY
        self.register_method(REMOVE_CAST_IDENTITY, self.remove_identity)
        self.register_method(FOLD_CAST, self.fold_cast)

    @staticmethod
    def remove_identity(node, graph, force_prune=True):
        # TODO Properly identify and remove casts once datatypes are trackable in IR
        if node.op.from_type == node.op.to_type or force_prune:
            graph.squash_identity(node)

    @staticmethod
    def fold_cast(graph):
        # scenario : one cast, one back to back cast type consumer.
        # in_tensor -> cast_0 -> cast_1 -> out_tensor
        # scenario_2 : one cast, two or more back to back cast type consumers with the same output dtype.
        # in_tensor -> cast_0 -> cast_1 -> out_tensor_1
        #                     -> cast_2 -> out_tensor_2
        sequence = [
                ("cast",
                    (),
                    ()
                ),
                ("cast",
                    ("MATCH_NUM_BUFS", [("cast", "ALL")]),
                    ()
                )
                ]

        matched_node_list = graph.get_matched_nodes(sequence)
        for node_tuple in matched_node_list:
            cast_node, _ = node_tuple
            cast_node_output_buf = graph.get_output_buffers(cast_node)[0]
            cast_op_name = cast_node.op.name
            cast_from_dtype = cast_node.op.from_type
            cast_node_input_names = cast_node.input_names

            # transform set to list
            consumers = list(cast_node_output_buf.consumers)
            if len(consumers) >= 1 and \
                all([consumer.op.type == op_adapter.CastOp.TRANSLATION_KEY for consumer in consumers]):
                # check if all the consumers' op type is cast then squash the first cast node in each node_tuple.
                graph.squash(cast_node, cast_node_input_names[0])
                # change the from_dtype of all the Casts in consumer list to the from_dtype of the squash Cast.
                for i in range(len(consumers)):
                    consumers[i].op.from_type = cast_from_dtype
                log_debug2("Squash Cast: {}".format(cast_op_name))


@register_layer_optimization
class OptimizeChannelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ChannelShuffleOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCollectRpnProposalsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CollectRpnProposalsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeColorTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ColorTransformOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            return True
        return False


@register_layer_optimization
class OptimizeConvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Conv2dOp.TRANSLATION_KEY
        self.register_method(PREPARE_BIASES, self.prepare_biases)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_conv2d_weights_dims(weights_buffer.shape)[-1]
        add_or_broadcast_bias(node, graph, output_channel)

    def prepare_inputs_as_params(self, node, graph):
        prepare_conv_inputs_as_params(graph, node)

    def axes_to_spatial_first_order(self, node, graph):
        if isinstance(graph.src_axis_order, (CaffeAxisOrder, SpatialLastAxisOrder)):
            input_buffers = graph.get_input_buffers(node)
            input_axis_formats = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NDHWC, transpose it to OIDHW by using a transpose to NCDHW. Then, to DHWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NDHWC,
                                         AxisTracker.AxisFormat.OIDHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 5):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCDHW, which is actually taking us back to OIDHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.OIDHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to DHWIO from OIDHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.DHWIO,
                                              graph.src_axis_order.permute_conv3d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.DHWIO

                # Update input_buffers and input_axis_formats after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_axis_formats = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NSC, transpose it to OIHW by using a transpose to NCS. Then, to HWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.OIHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 4):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCS, which is actually taking us back to OIHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.OIHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to HWIO from OIHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.HWIO,
                                              graph.src_axis_order.permute_conv2d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.HWIO

                # Update input_buffers and input_axis_formats after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_axis_formats = [buf.axis_format for buf in input_buffers]

            if any(axis_format in input_axis_formats for axis_format in [AxisTracker.AxisFormat.NDHWC,
                                                                         AxisTracker.AxisFormat.DHWIO,
                                                                         AxisTracker.AxisFormat.NSC,
                                                                         AxisTracker.AxisFormat.HWIO,
                                                                         AxisTracker.AxisFormat.ANY,
                                                                         AxisTracker.AxisFormat.NONTRIVIAL]):
                AxisTracker.image_to_channel_last_order(node, graph)
                output_buffer = graph.get_output_buffers(node)[0]
                # image_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
                # Enforce the output format here according to output buffer's rank
                output_buffer.axis_format = AxisOrder().get_axis_format(output_buffer.rank())
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_CONVOLUTION_UNEXPECTED_INPUT_ORDER")
                                 (input_axis_formats))
            return True


@register_layer_optimization
class OptimizeConvolution1DTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.Conv1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        conv_op_name = node.op.name + "_2d"
        self.nn_2d_op = op_adapter.Conv2dOp(conv_op_name,
                                            bias_op_name=node.op.bias_op_name,
                                            pady_before=node.op.pad_amount[0],
                                            pady_after=node.op.pad_amount[1],
                                            padx_before=0,
                                            padx_after=0,
                                            padding_size_strategy=node.op.padding_size_strategy,
                                            stridex=1,
                                            stridey=node.op.stride[0],
                                            dilationx=1,
                                            dilationy=node.op.dilation[0],
                                            groups=node.op.group,
                                            data_layout=AxisTracker.AxisFormat.NCS)
        super().expand_1d_spatial_nn_nodes(node, graph)


@register_layer_optimization
class OptimizeConvolution3DTranslation(OptimizeConvolutionTranslation):
    def __init__(self):
        OptimizeConvolutionTranslation.__init__(self)
        self.op_type = op_adapter.Conv3dOp.TRANSLATION_KEY

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_conv3d_weights_dims(weights_buffer.shape)[-1]
        add_or_broadcast_bias(node, graph, output_channel)


@register_layer_optimization
class OptimizeConcatTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConcatOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)

    def axes_to_spatial_first_order(self, node, graph):
        ret = super().axes_to_spatial_first_order(node, graph)
        if not ret:
            # If ret is False, no change happened in super(), no futher action is needed, so return
            return ret
        buf = graph.get_buffer(node.output_names[0])

        # permute axis if input is permuted
        input_axis_formats = graph.get_input_axis_formats(node)
        # assert that axis formats of all inputs match
        first_in_format = input_axis_formats[0]
        if not all([in_format == first_in_format for in_format in input_axis_formats]):
            input_bufs = graph.get_input_buffers(node)
            input_ranks = [input_buf.rank() for input_buf in input_bufs]
            first_input_rank = input_ranks[0]
            if not all([input_rank == first_input_rank for input_rank in input_ranks]):
                raise ValueError("ranks of all inputs are not matched: {}".format(input_ranks))
            elif AxisTracker.AxisFormat.NONTRIVIAL not in input_axis_formats:
                raise ValueError("axis formats of all inputs are not matched: {}".format(input_axis_formats))
            else:
                for input_name in node.input_names:
                    input_buf = graph.get_buffer(input_name)
                    if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
                        graph.inject_implicit_permute(
                            input_buf.name,
                            AxisTracker.AxisFormat.NONTRIVIAL,
                            spatial_first_format_to_channel_first_permute_order[input_buf.axis_format],
                            consumers=[node.op.name]
                        )
                    else:
                        # for NONTRIVIAL, ANY, NC, NF, and channel first orders
                        # we should directly pass without modification
                        pass
                # after aligning all input axis formats, refresh the input_axis_formats variable
                input_axis_formats = graph.get_input_axis_formats(node)
                # set output buffer's axis format to NONTRIVIAL
                output_buf = graph.get_output_buffers(node)[0]
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        if input_axis_formats != node.op.data_axis_formats:
            if buf.axis_format == AxisTracker.AxisFormat.NTF:
                axis_map = graph.src_axis_order.permute_time_series_to_ir
            else:
                axis_map = graph.src_axis_order.permute_sequence_from_ir[buf.rank() - 1]
            node.op.axis = axis_map[node.op.axis]
        # shape assertion
        ref_inp_shape = graph.get_buffer(node.input_names[0]).shape
        for input_buf in graph.get_input_buffers(node)[1:]:
            inp_shape = input_buf.shape
            for i in range(len(ref_inp_shape)):
                if i != node.op.axis and ref_inp_shape[i] != inp_shape[i]:
                    raise ValueError("input shapes of concat op not aligned: {}, {} while axis = {}"
                                     .format(ref_inp_shape, inp_shape, node.op.axis))
        return True

    @staticmethod
    def fold_concats(graph):
        def validate_concat_axis(nodes_tuple):
            concat_node_ = nodes_tuple[0]
            concat_node_input_bufs_ = graph.get_input_buffers(concat_node_)
            for buf_ in concat_node_input_bufs_:
                if buf_.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_node_ = buf_.producer
                    # only fold concats with same axis
                    if prev_concat_node_.op.axis != concat_node_.op.axis:
                        log_debug2("Found concat node({}) with a concat input, but axis does not match for input ({}), "
                                   "{} != {} ", concat_node_.op.name, prev_concat_node_.op.name,
                                   prev_concat_node_.op.axis, concat_node_.op.axis)
                        return False

            return True

        sequence = [
                    ("Concat",
                     ("FLEXIBLE_NUM_BUFS", [("Concat", "ANY")]),
                     ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_concat_axis)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_node_input_bufs = graph.get_input_buffers(concat_node)

            for buf in concat_node_input_bufs:
                if buf.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_buf = buf  # for readability
                    prev_concat_node = prev_concat_buf.producer

                    # remove prev concat as input from current concat and replace with prev concat's input names
                    prev_concat_inputs = prev_concat_node.input_names
                    idx = concat_node.input_names.index(prev_concat_buf.name)
                    concat_node.input_names.remove(prev_concat_buf.name)
                    # extend the inputs in the same index as prev concat
                    concat_node.input_names[idx:idx] = prev_concat_inputs
                    # update the concat.op.data_axis_formats since inputs are updated
                    concat_node.op.data_axis_formats.pop(idx)
                    concat_node.op.data_axis_formats[idx:idx] = prev_concat_node.op.data_axis_formats

                    prev_concat_buf.consumers.remove(concat_node)

                    # we can prune the prev concat node if the current concat was the only consumer.
                    if len(prev_concat_buf.consumers) == 0:
                        graph.prune(prev_concat_node)

                    # remove prev concat as consumer for prev concat's input bufs and replace with current concat
                    for input_name in prev_concat_inputs:
                        input_buf = graph.get_buffer(input_name)
                        input_buf.consumers.add(concat_node)

                    log_debug2(code_to_message.get_debugging_message("DEBUG_CONCAT_FOLD")(prev_concat_node.op.name,
                                                                                          concat_node.op.name))


@register_layer_optimization
class OptimizeConstantTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConstantOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_buffer(node.output_names[0])
        output_rank = output_buf.rank()

        # TODO Remove this code once limitations of AxisTracking are resolved
        # If the consumer of this buffer has another input with NSC format, and this buffer is 3D, it needs to be
        # padded with a 1 and have its constant operation permuted
        consumers = list(output_buf.consumers)
        if len(consumers) and output_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            consumer_has_dimension_mismatch = [False] * len(consumers)
            for i, consumer in enumerate(consumers):
                if not isinstance(consumer.op, op_adapter.ElementwiseBinaryOp):
                    continue
                for input_buffer in graph.get_input_buffers(consumer):
                    if input_buffer.axis_format == AxisTracker.AxisFormat.NDHWC and output_rank == 4:
                        consumer_has_dimension_mismatch[i] = True
                        break
                    elif input_buffer.axis_format == AxisTracker.AxisFormat.NSC and output_rank == 3:
                        consumer_has_dimension_mismatch[i] = True
                        break

            if all(consumer_has_dimension_mismatch):
                log_debug("All consumers of {} node {} have {}D-{}D rank mismatch in inputs. Updating buffer {}.".format(
                    node.op.type, node.op.name, output_rank+1, output_rank, output_buf.name))
                # Capture tensor and prepare for placement in graph
                const_tensor = output_buf.producer.op.tensor
                const_tensor_shape = [1, *list(const_tensor.shape)]
                const_tensor = np.reshape(const_tensor, const_tensor_shape)
                # Modify the graph according to updated shape
                output_buf.producer.op.tensor = const_tensor
                output_buf.shape = const_tensor_shape
                output_buf.axis_format = graph.src_axis_order.get_axis_format(output_rank+1)
            elif any(consumer_has_dimension_mismatch):
                # Remove consumers that need to be updated from current graph
                consumers_to_update = [consumer for i, consumer in output_buf.consumers if
                                       consumer_has_dimension_mismatch[i]]
                for consumer in consumers_to_update:
                    consumer.input_names.remove(output_buf.name)
                    output_buf.remove(consumer)
                # Create the new constant tensor
                const_tensor = output_buf.producer.op.tensor
                const_tensor_shape = [1, *list(const_tensor.shape)]
                const_tensor = np.reshape(const_tensor, const_tensor_shape)
                # Create the new N+1D constant operation
                const_op_name = output_buf.name + "_{}d".format(output_rank+1)
                const_op = op_adapter.ConstantOp(const_op_name, const_tensor,
                                                 quantizable=output_buf.producer.op.quantizable)
                # Place the new N+1D constant operation in graph
                log_debug("At least one, but not all consumers of buffer {} have {}D-{}D dimension mismatch. Creating "
                          "a new constant {}D constant operation named {}."
                          .format(output_buf.name, output_rank+1, output_rank, output_rank+1, const_op_name))
                graph.add(const_op, [], [const_op_name], axis_formats=[graph.src_axis_order.get_axis_format(output_rank+1)])
                graph.get_buffer(const_op_name).consumers = consumers_to_update
                for consumer in consumers_to_update:
                    consumer.input_names.add(const_op_name)

        # Permute the constant data if necessary
        if output_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.NCDHW_TO_NDHWC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
            output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
        elif output_buf.axis_format == AxisTracker.AxisFormat.NCS:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            output_buf.axis_format = AxisTracker.AxisFormat.NSC
        elif output_buf.axis_format == AxisTracker.AxisFormat.NCF:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.NCF_TO_NFC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCF_TO_NFC)
            output_buf.axis_format = AxisTracker.AxisFormat.NFC
        elif output_buf.axis_format == AxisTracker.AxisFormat.TNF:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.TNF_TO_NTF))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            output_buf.axis_format = AxisTracker.AxisFormat.NTF
        elif output_buf.axis_format == AxisTracker.AxisFormat.OIDHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.OIDHW_TO_DHWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.OIDHW_TO_DHWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.DHWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.IODHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.IODHW_TO_DHWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.IODHW_TO_DHWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.DHWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.OIHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.OIHW_TO_HWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.OIHW_TO_HWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.HWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.IOHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.IOHW_TO_HWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.IOHW_TO_HWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.HWIO

        return True

    @staticmethod
    def remove_identity(node, graph):
        # Prune this node if it's an input to a weight layer and was used internally
        if getattr(graph, "weights", None) and getattr(graph.weights, "consumed", None) \
                and graph.weights.consumed(node.output_names[0]):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONSTANT_PRUNED")(node.output_names[0]))
            graph.prune(node)


@register_layer_optimization
class OptimizeConvertTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvertOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCumSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CumSumOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeCumSumTranslation, self).axes_to_spatial_first_order(node, graph):
            return False

        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            node.op.axis = AxisTracker.AxisFormat.NCDHW_TO_NDHWC[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCDHW and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC:
            node.op.axis = AxisTracker.AxisFormat.NDHWC_TO_NCDHW[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            node.op.axis = AxisTracker.AxisFormat.NCS_TO_NSC[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCS and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NSC:
            node.op.axis = AxisTracker.AxisFormat.NSC_TO_NCS[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            node.op.axis = AxisTracker.AxisFormat.NCF_TO_NFC[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NFC:
            node.op.axis = AxisTracker.AxisFormat.NFC_TO_NCF[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            node.op.axis = AxisTracker.AxisFormat.TNF_TO_NTF[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.TNF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NTF:
            node.op.axis = AxisTracker.AxisFormat.NTF_TO_TNF[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NONTRIVIAL:
            pass
        else:
            raise ValueError("Unexpected input buffer axis format: {}, for {} Op".format(input_buf.axis_format, node.op.name))

        return True


@register_layer_optimization
class OptimizeCustomOpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CustomOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Todo: revisit and modify when the layout support for CustomOps is added [AISW-55482].
        ret = super(OptimizeCustomOpTranslation, self).axes_to_spatial_first_order(node, graph)

        return ret


@register_layer_optimization
class OptimizeTransposeConv1dTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.TransposeConv1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        conv_op_name = node.op.name + "_2d"
        self.nn_2d_op = op_adapter.TransposeConv2dOp(conv_op_name,
                                                     bias_op_name=node.op.bias_op_name,
                                                     stridex=1,
                                                     stridey=node.op.stride[0],
                                                     padx_before=0,
                                                     padx_after=0,
                                                     pady_before=node.op.pad_amount[0],
                                                     pady_after=node.op.pad_amount[1],
                                                     output_paddingx=0,
                                                     output_paddingy=node.op.output_padding,
                                                     padding_size_strategy=node.op.padding_size_strategy,
                                                     output_height=node.op.output_size,
                                                     output_width=1,
                                                     groups=node.op.group)
        super().expand_1d_spatial_nn_nodes(node, graph)

@register_layer_optimization
class OptimizeTransposeConv2dTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TransposeConv2dOp.TRANSLATION_KEY
        self.register_method(PREPARE_BIASES, self.prepare_biases)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        if isinstance(graph.src_axis_order, (CaffeAxisOrder, SpatialLastAxisOrder)):
            input_buffers = graph.get_input_buffers(node)
            input_axis_formats = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NDHWC, transpose it to IODHW by using a transpose to NCDHW. Then, to DHWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NDHWC,
                                         AxisTracker.AxisFormat.IODHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 5):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCDHW, which is actually taking us back to IODHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.IODHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to DHWIO from IODHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.DHWIO,
                                              graph.src_axis_order.permute_deconv3d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.DHWIO

                # Update input_buffers and input_orders after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_orders = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NSC, transpose it to IOHW by using a transpose to NCS. Then, to HWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.IOHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 4):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCS, which is actually taking us back to IOHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.IOHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to HWIO from IOHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.HWIO,
                                              graph.src_axis_order.permute_deconv2d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.HWIO

                # Update input_buffers and input_orders after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_orders = [buf.axis_format for buf in input_buffers]

            if any(format in input_axis_formats for format in [AxisTracker.AxisFormat.NDHWC,
                                                               AxisTracker.AxisFormat.DHWIO,
                                                               AxisTracker.AxisFormat.NSC,
                                                               AxisTracker.AxisFormat.HWIO,
                                                               AxisTracker.AxisFormat.ANY,
                                                               AxisTracker.AxisFormat.NONTRIVIAL]):
                AxisTracker.image_to_channel_last_order(node, graph)
                output_buffer = graph.get_output_buffers(node)[0]
                # image_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
                # Enforce the output format here according to output buffer's rank
                output_buffer.axis_format = AxisOrder().get_axis_format(output_buffer.rank())
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_TRANPOSE_CONV_UNEXPECTED_INPUT_ORDER")
                                 (input_orders))

            return True

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_deconv2d_weights_dims(weights_buffer.shape)[-1] * node.op.group
        add_or_broadcast_bias(node, graph, output_channel)

    def prepare_inputs_as_params(self, node, graph):
        prepare_conv_inputs_as_params(graph, node)


@register_layer_optimization
class OptimizeDepthwiseConvolution1DTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.DepthwiseConv1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        conv_op_name = node.op.name + "_2d"
        self.nn_2d_op = op_adapter.DepthwiseConv2dOp(conv_op_name,
                                                    bias_op_name=node.op.bias_op_name,
                                                    pady_before=node.op.pad_amount[0],
                                                    pady_after=node.op.pad_amount[1],
                                                    padx_before=0,
                                                    padx_after=0,
                                                    padding_size_strategy=node.op.padding_size_strategy,
                                                    stridex=1,
                                                    stridey=node.op.stride[0],
                                                    dilationx=1,
                                                    dilationy=node.op.dilation[0],
                                                    data_layout=AxisTracker.AxisFormat.NCS)
        super().expand_1d_spatial_nn_nodes(node, graph)


@register_layer_optimization
class OptimizeDepthwiseConvolutionTranslation(OptimizeConvolutionTranslation):
    def __init__(self):
        OptimizeConvolutionTranslation.__init__(self)
        self.op_type = op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_conv2d_weights_dims(weights_buffer.shape)[-1]
        add_or_broadcast_bias(node, graph, output_channel)


@register_layer_optimization
class OptimizeDetectionOutTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DetectionOutputOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)
        self.register_method(MATCH_CAFFE_SSD_TO_TF, self.caffe_ssd_to_tf)

    @staticmethod
    def fold_concats(graph):
        def process_ssd_priorbox_concat_layer(input_buffers_):
            concatenated_priorbox_data = []
            concatenated_priorbox_cz_data = []
            concatenated_priorbox_variance = []
            scale_factors_ = input_buffers_[0].producer.op.scale_factors
            for input_buffer in input_buffers_:
                priorbox_op = input_buffer.producer.op
                concatenated_priorbox_data.extend(priorbox_op.priorbox_box_output[0])
                concatenated_priorbox_variance.extend(priorbox_op.priorbox_box_output[1])
                concatenated_priorbox_cz_data.extend(priorbox_op.priorbox_box_cz_output)
                if scale_factors_ != priorbox_op.scale_factors:
                    # Currently only support 1 set of scale factor for priorboxes.
                    raise ValueError(code_to_message.get_error_message("ERROR_INVALID_PRIORBOX_VARIANCES")
                                     (scale_factors_, input_buffers_[0].producer.op.name,
                                      priorbox_op.scale_factors, priorbox_op.name))

            return concatenated_priorbox_data + concatenated_priorbox_variance, concatenated_priorbox_cz_data, \
                   scale_factors_

        sequence = [
            ("Concat",
                ("FLEXIBLE_NUM_BUFS", [("identity", "ALL")]),  # identity here since all priorboxes are mapped to IdentityOp
                ("MATCH_NUM_BUFS", [("DetectionOutput", "ALL")])
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_input_buffers = graph.get_input_buffers(concat_node)
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            detection_out_node = concat_output_buffer.consumers.pop()
            priorbox_data, priorbox_cz_data, scale_factors = process_ssd_priorbox_concat_layer(concat_input_buffers)
            detection_out_node.op.priorbox_data = priorbox_data
            detection_out_node.op.priorbox_center_size_data = priorbox_cz_data
            # order determined per caffe/util/bbox_util.cpp
            delta_scaling_factors = np.array([
                scale_factors[0],
                scale_factors[1],
                scale_factors[2],
                scale_factors[3]
            ], dtype=np.float32)
            detection_out_node.op.delta_scaling_factors = delta_scaling_factors

            # remove concat node.
            detection_out_node.input_names.remove(concat_output_buffer.name)
            graph.prune(concat_node)

            # remove priorboxes
            for buf in concat_input_buffers:
                graph.prune(buf.producer)

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_FOLDING")(concat_node.op.name,
                                                                                           detection_out_node.op.name))

    @staticmethod
    def caffe_ssd_to_tf(graph):
        sequence = [
            ("DetectionOutput",
                ("MATCH_NUM_BUFS", [("Reshape", "ANY"), ("Concat", "ANY")]),  # flattened scores and boxes
                ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            detection_out_node = node_tuple[0]
            for input_name in detection_out_node.input_names:
                node = graph.get_producer_node(input_name)
                if node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                    reshape_node = node
                elif node.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    concat_node = node
                else:
                    raise ValueError(code_to_message.get_error_message("ERROR_DETECTIONOUT_UNKNOWN_INPUTS")
                                     (node.op.type))

            # 0. Verify valid anchors/priorboxes
            log_assert(detection_out_node.op.code_type == op_adapter.DetectionOutputOp.PriorBoxType.CENTER_SIZE,
                       "DetectionOut Op only supports center size code type. Got {}".
                       format(detection_out_node.op.code_type))

            # 1. Pre-process steps
            # Caffe score input is flattened, remove reshape to match shape [batch, num_anchors, num_classes]
            reshape_output_buffer = graph.get_output_buffers(reshape_node)[0]
            detection_out_node.input_names.remove(reshape_output_buffer.name)
            detection_out_node.input_names.insert(0, reshape_node.input_names[0])
            graph.get_buffer(reshape_node.input_names[0]).consumers.add(detection_out_node)

            reshape_output_buffer.consumers.remove(detection_out_node)
            # remove reshape node if applicable.
            if len(reshape_output_buffer.consumers) == 0:
                graph.prune(reshape_node)

            # Caffe boxes(location) data is also flattened. Reshape to [batch, num_boxes, 4]
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            concat_buf_shape = concat_output_buffer.shape
            # add reshape node
            reshape_name = concat_node.op.name + "_preprocess_reshape"
            reshape_op = op_adapter.ReshapeOp(reshape_name, shape=[concat_buf_shape[0],
                                                                   int(concat_buf_shape[1] / 4),
                                                                   4])
            graph.inject(reshape_op, input_name=concat_node.output_names[0], output_name=reshape_name,
                         consumer_names=detection_out_node.output_names)

            # DetectionOut in IR has priorboxes as param, need to add those to input instead
            detection_out_name = detection_out_node.op.name
            detection_out_node_idx = graph.nodes_in_order.index(detection_out_node)
            prior_box_name = detection_out_name + "_anchors"
            pbox_data = np.asarray(detection_out_node.op.priorbox_center_size_data, dtype=np.float32)\
                        .reshape(int(len(detection_out_node.op.priorbox_center_size_data)/4), 4)
            prior_box_op = op_adapter.ConstantOp(name=prior_box_name, tensor=pbox_data)
            graph.add(prior_box_op, input_names=[], output_names=[prior_box_name], idx=detection_out_node_idx-1)
            detection_out_node.input_names.append(prior_box_name)

            # Caffe Ssd scales is the reciprocal compared to TF scales
            detection_out_node.op.delta_scaling_factors = np.array([
                1 / detection_out_node.op.delta_scaling_factors[0],
                1 / detection_out_node.op.delta_scaling_factors[1],
                1 / detection_out_node.op.delta_scaling_factors[2],
                1 / detection_out_node.op.delta_scaling_factors[3],
            ], dtype=np.float32)
            # 2. Change DetectionOut's single output to multiple. Outputs:
            #    Expected: scores[1, max_num_det], boxes[1, max_num_det, 4], classes[1, max_num_det], num_det[batch],
            #    Caffe Style: 1 output of shape [1, 1, max_num_det, 7]
            #                   7(last dim above): [image_batch, label, confidence, x_min, y_min, x_max, y_max]
            detection_out_buf = graph.get_buffer(detection_out_node.output_names[0])
            boxes_shape = [detection_out_buf.shape[0], detection_out_node.op.keep_top_k, 4]  # [batch, max_num_detections, 4)
            boxes_name = detection_out_name + "_boxes"
            boxes_buf = op_graph.Buffer(boxes_name, boxes_shape, detection_out_node)
            graph.buffers[boxes_name] = boxes_buf

            scores_name = detection_out_name + "_scores"
            scores_buf = op_graph.Buffer(scores_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[scores_name] = scores_buf

            classes_name = detection_out_name + "_classes"
            classes_buf = op_graph.Buffer(classes_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[classes_name] = classes_buf

            num_det_name = detection_out_name + "_num_detections"
            num_det_buf = op_graph.Buffer(num_det_name, [boxes_shape[0]], detection_out_node)
            graph.buffers[num_det_name] = num_det_buf

            del graph.buffers[detection_out_node.output_names[0]]
            detection_out_node.output_names = [scores_name, boxes_name, classes_name, num_det_name]

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_CAFFE_TO_TF_STYLE")
                       (detection_out_node.op.name))

@register_layer_optimization
class OptimizeDequantizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DequantizeOp.TRANSLATION_KEY
        self.register_method(REMOVE_QUANT_NODES, self.remove_quant_nodes)

    @staticmethod
    def remove_quant_nodes(node, graph):
        graph.squash(node, input_name=node.input_names[0])
        log_debug("Remove dequantize op {}".format(node.op.name))


@register_layer_optimization
class OptimizeDistributeFpnProposalsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DistributeFpnProposalsOp.TRANSLATION_KEY


class OptimizeElementwiseTranslation(OptimizationTranslationBase):
    def __init__(self, op_type):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_type

    def axes_to_spatial_first_order(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        # Ensure:
        #   1. if all inputs have matching ranks, formats should also match and can be changed to IR
        #   2. if ranks don't match, no axes transformation should happen
        def matching_ranks(input_buffers):
            if all(map(lambda buf: buf.rank() == input_buffers[0].rank() or
                                   buf.rank() == 1 and buf.shape[0] == 1,
                       input_buffers)):
                return True
            else:
                return False

        input_buffers = graph.get_input_buffers(node)
        if matching_ranks(input_buffers) and AxisTracker.AxisFormat.NONTRIVIAL not in graph.get_input_axis_formats(node):
            # Matching ranks makes the tensors broadcastable even in IR
            # Axis Tracker function will enforce IR order
            return super().axes_to_spatial_first_order(node, graph)
        else:
            # If not matching ranks, permute to IR will break broadcast capability
            for i, buf in enumerate(input_buffers):
                # this should only work for buffers which has rank greater than 2
                if node.op.data_axis_formats[i] != buf.axis_format and \
                        buf.axis_format in AxisOrder().axis_formats and buf.rank() > 2:
                    # Transpose to maintain src format
                    graph.inject_implicit_permute(
                        buf.name,
                        spatial_first_format_to_channel_first_format[buf.axis_format],
                        spatial_first_format_to_channel_first_permute_order[buf.axis_format],
                        [node.op.name]
                    )
            return False


@register_layer_optimization
class OptimizeElementwiseAndTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_AND])


@register_layer_optimization
class OptimizeElementwiseDivTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE])
        self.register_method(SQUASH_DIV, self.squash_div)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    @staticmethod
    def squash_div(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0],
                op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])

    @staticmethod
    def remove_identity(node, graph):
        divisor_op = graph.get_buffer(node.input_names[1]).producer.op
        # squash the op if the divisor is a tensor of all ones
        if divisor_op.type == "constant" and np.all(divisor_op.tensor == 1):
            try:
                graph.squash(node, node.input_names[0])
            except RuntimeError as e:
                log_debug("Squash elementwise div op {} due to identity not possible ".format(node.op.name))

@register_layer_optimization
class OptimizeElementwiseEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EQUAL])


@register_layer_optimization
class OptimizeElementwiseFloorDivTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV])


@register_layer_optimization
class OptimizeElementwiseFmodTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FMOD])


@register_layer_optimization
class OptimizeElementwiseGreaterTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER])


@register_layer_optimization
class OptimizeElementwiseGreaterEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_GREATER_EQUAL])


@register_layer_optimization
class OptimizeElementwiseLessTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS])


@register_layer_optimization
class OptimizeElementwiseLessEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LESS_EQUAL])


@register_layer_optimization
class OptimizeElementwiseMaxTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM])


@register_layer_optimization
class OptimizeElementwiseMinTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM])


@register_layer_optimization
class OptimizeElementwiseModTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MOD])


@register_layer_optimization
class OptimizeElementwiseNotEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NOT_EQUAL])


@register_layer_optimization
class OptimizeElementwisePowerTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_POWER])
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def prepare_inputs_as_params(self, node, graph):
        exponent_buffer = graph.get_buffer(node.input_names[1])
        exponent_node = exponent_buffer.producer
        if exponent_node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY:
            raise ValueError("Dynamic exponents on node {} are not supported in this backend.".format(node.op.name))
        node.op.power = exponent_node.op.tensor
        graph.remove_node_as_consumer(node, exponent_buffer.name)


@register_layer_optimization
class OptimizeElementwiseProductTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY])
        self.register_method(SQUASH_PROD, self.squash_prod)

    @staticmethod
    def squash_prod(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseSelectTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseTernaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SELECT]


@register_layer_optimization
class OptimizeElementwiseSubTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT])
        self.register_method(SQUASH_SUB, self.squash_sub)

    @staticmethod
    def squash_sub(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseSumTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD])
        self.register_method(SQUASH_SUM, self.squash_sum)

    @staticmethod
    def squash_sum(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseOrTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_OR])


@register_layer_optimization
class OptimizeElementwiseUnaryAbsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ABS]


@register_layer_optimization
class OptimizeElementwiseUnaryAsinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ASIN]


@register_layer_optimization
class OptimizeElementwiseUnaryAtanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ATAN]


@register_layer_optimization
class OptimizeElementwiseUnaryCeilTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_CEIL]


@register_layer_optimization
class OptimizeElementwiseUnaryCosTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_COS]


@register_layer_optimization
class OptimizeElementwiseUnaryExpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_EXP]


@register_layer_optimization
class OptimizeElementwiseUnaryFloorTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_FLOOR]


@register_layer_optimization
class OptimizeElementwiseUnaryLogTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_LOG]


@register_layer_optimization
class OptimizeElementwiseUnaryNegTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NEG]

    @staticmethod
    def optimize_negation(graph):
        def validate_neg(nodes_tuple):
            for input_name_ in nodes_tuple[0].input_names:
                node_ = graph.get_producer_node(input_name_)
                if node_.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                        all(val == -1 for val in np.array(node_.op.tensor).flatten()):
                    return True

            return False

        # Optimization: -1 * A => Neg(A)
        sequences = [
            [
                ("elementwise_product",
                 ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_neg)
            for node_tuple in matched_node_list:
                prod_node = node_tuple[0]
                non_neg_input_node = None
                neg_const_input_node = None
                for input_name in prod_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    if input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                            all(val == -1 for val in np.array(input_node.op.tensor).flatten()):
                        neg_const_input_node = input_node
                    else:
                        non_neg_input_node = input_node
                neg_const_input_buf = graph.get_buffer(neg_const_input_node.output_names[0])
                non_neg_input_buf = graph.get_buffer(non_neg_input_node.output_names[0])

                if non_neg_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    input_tensor = non_neg_input_node.op.tensor
                    output_tensor = np.negative(input_tensor).astype(input_tensor.dtype)
                    # remove all input of prod to replace with constant node
                    prod_node.input_names = []
                    neg_const_input_buf.consumers.remove(prod_node)
                    non_neg_input_buf.consumers.remove(prod_node)

                    # replace prod with const
                    const_op = op_adapter.ConstantOp(prod_node.op.name, tensor=output_tensor)
                    graph.replace(prod_node.op, const_op)
                    log_debug2("Optimization of -1 * const(A) => Const(B)  complete. Op {} replaced with ConstOp"
                               .format(prod_node.op.name))
                else:
                    # remove const as input to prod, the prod node will then be replaced as Neg
                    neg_const_input_buf.consumers.remove(prod_node)
                    prod_node.input_names.remove(neg_const_input_node.output_names[0])

                    neg_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.ElementwiseUnaryOp.type,
                                                                          op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NEG])
                    neg_op = op_adapter.ElementwiseUnaryOp(neg_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_NEG)
                    graph.replace(prod_node.op, neg_op)
                    log_debug2("Optimization of -1 * A => Neg(A) complete. Op {} replaced with NegOp"
                               .format(prod_node.op.name))

                if len(neg_const_input_buf.consumers) == 0:
                    graph.prune(neg_const_input_node)
                if len(non_neg_input_buf.consumers) == 0:
                    graph.prune(non_neg_input_node)

        # Optimization: A + Neg(B) => A - B
        #               Neg(A) + B => B - A
        #               Neg(A) + Neg(B) => Neg(A) - B
        sequences = [
            [
                ("elementwise_sum",
                 ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_neg", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence)
            for node_tuple in matched_node_list:
                sum_node = node_tuple[0]
                neg_node_to_prune = None
                for input_name in sum_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    input_buf = graph.get_buffer(input_name)
                    if input_node.op.type == op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NEG]:
                        # if more than consumer of NegOp then we cant remove it hence optimization
                        # is not really relevant.
                        if len(input_buf.consumers) == 1:
                            neg_node_to_prune = input_node

                if neg_node_to_prune is not None:
                    # Update the input and consumer list and remove NegOp from graph
                    neg_idx = sum_node.input_names.index(neg_node_to_prune.output_names[0])
                    sum_input_names = sum_node.input_names[:]
                    neg_input_name = neg_node_to_prune.input_names[0]
                    neg_input_buf = graph.get_buffer(neg_input_name)
                    graph.prune(neg_node_to_prune, force_remove=True)
                    if neg_idx == 0:
                        # got Neg(A) + B, need B - A
                        sum_input_names[0] = sum_input_names[1]
                        sum_input_names[1] = neg_input_name
                    else:
                        # Neg(A) + Neg(B) or A + Neg(B)
                        sum_input_names[neg_idx] = neg_input_name
                    neg_input_buf.consumers.add(sum_node)
                    sum_node.input_names = sum_input_names

                    op_type = ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT
                    legacy_op_type = op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[op_type]
                    sub_op_name = graph.naming_policy.get_op_name_by_type(op_type, legacy_op_type)
                    sub_op = op_adapter.ElementwiseBinaryOp(sub_op_name,
                                                            eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT)

                    graph.replace(sum_node.op, sub_op)
                    log_debug2("Optimization of addition to a negative of an op (e.g: A + Neg(B) => A - B) complete. "
                               "Op {} replaced with SubOp"
                               .format(sum_node.op.name))


@register_layer_optimization
class OptimizeElementwiseUnaryNotTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_NOT]


@register_layer_optimization
class OptimizeElementwiseUnaryRoundTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ROUND]


@register_layer_optimization
class OptimizeElementwiseUnaryRsqrtTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_RSQRT]


@register_layer_optimization
class OptimizeElementwiseUnarySignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SIGN]


@register_layer_optimization
class OptimizeElementwiseUnarySinTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SIN]


@register_layer_optimization
class OptimizeElementwiseUnarySoftplusTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SOFTPLUS]


@register_layer_optimization
class OptimizeElementwiseUnarySqrtTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseUnaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SQUARE_ROOT]


@register_layer_optimization
class OptimizeErfTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ErfOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExpandTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExpandOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    @staticmethod
    def remove_identity(node, graph):
        input_shape = graph.get_input_shapes(node)[0]
        output_shape = graph.get_output_shapes(node)[0]
        if input_shape == output_shape:
            graph.squash(node, input_name=node.input_names[0])


@register_layer_optimization
class OptimizeFullyConnectedTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FullyConnectedOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)
        self.register_method(SQUASH_SUM, self.squash_sum)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.log_axes_transformation(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            if input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
                AxisTracker.enforce_input_axis_format(graph, input_buf.name, AxisTracker.AxisFormat.NSC,
                                                      AxisTracker.AxisFormat.NCS_TO_NSC)

                # weights axis_format will be set to NONTRIVIAL after transpose
                # to avoid transposing shared weights multiple times
                weights_buf = graph.get_buffer(node.input_names[1])
                if weights_buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL and not isinstance(graph.src_axis_order, RelayAxisOrder):
                    # weights expect NCHW order, need to permute
                    input_buf = graph.get_input_buffers(node)[0]
                    batch, height, width, channel = input_buf.shape
                    weights = weights_buf.producer.op.tensor

                    # Assuming FC: W^Tx + b and weights have shape (input_size, output_size)
                    input_size = weights.shape[0]
                    output_size = weights.shape[1]
                    log_assert(input_size == channel * height * width,
                               code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                              (input_size, output_size),
                                                                                              (batch,  height, width, channel)))

                    weights.shape = (channel, height, width, output_size)
                    weights = np.transpose(weights, (3, 1, 2, 0))
                    weights = np.ascontiguousarray(weights, dtype=np.float32)
                    weights.shape = (output_size, input_size)
                    weights_buf.producer.op.tensor = weights
                    weights_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                    node.op.transpose_b = False
                    # since weights are getting transposed here, the info should be updated in transpose_b
                    # showing no further transpose is required
        else:
            # weights axis_format will be set to NONTRIVIAL after transpose
            # to avoid transposing shared weights multiple times
            weights_buf = graph.get_buffer(node.input_names[1])
            if weights_buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL and not isinstance(graph.src_axis_order, RelayAxisOrder):
                # again, need to transpose weights for spatial_first order
                weights = weights_buf.producer.op.tensor
                weights = np.ascontiguousarray(np.transpose(weights, (1, 0)))
                weights_buf.producer.op.tensor = weights
                weights_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                node.op.transpose_b = False

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.NF

        return True

    @staticmethod
    def squash_batchnorm(graph):

        sequence1 = [
            ("FullyConnected",
                ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
                ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
             )
        ]

        sequence2 = [
            ("FullyConnected",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
               ("MATCH_NUM_BUFS", [("Reshape", "ANY")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("FullyConnected", "ANY")]),
               ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
            )
        ]

        sequences = [sequence1, sequence2]
        for idx, sequence in enumerate(sequences):

            matched_node_list = graph.get_matched_nodes(sequence)

            for node_tuple in matched_node_list:
                # sanity check
                log_assert(len(node_tuple) == len(sequence),
                        "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                        len(node_tuple), len(sequence))

                fc_node = node_tuple[0]
                if(idx == 0):
                    bn_node = next(iter(graph.get_output_buffers(fc_node)[0].consumers))
                else:
                    reshape_node = node_tuple[1]
                    bn_node = next(iter(graph.get_output_buffers(reshape_node)[0].consumers))
                fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                bn_node_weights =  graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
                bn_node_bias =  graph.get_buffer(bn_node.input_names[2]).producer.op.tensor
                bn_input_buffer = graph.get_input_buffers(bn_node)[0]
                bn_output_buffer = graph.get_output_buffers(bn_node)[0]
                manage_shared_static_input(graph, fc_node, 1)
                weights = graph.get_buffer(fc_node.input_names[1]).producer.op.tensor
                manage_shared_static_input(graph, fc_node, 2)
                bias = graph.get_buffer(fc_node.input_names[2]).producer.op.tensor
                broadcasted_tensor = np.zeros(len(bn_node_weights), dtype=np.float32)
                if not fc_node.op.transpose_b:
                    weight_tensor = np.transpose(weights, (1, 0)).copy()
                else:
                    weight_tensor = weights.copy()
                broadcasted_tensor = broadcasted_tensor + weight_tensor
                broadcasted_tensor = broadcasted_tensor * bn_node_weights
                if not fc_node.op.transpose_b:
                    broadcasted_transpose = np.transpose(broadcasted_tensor, (1, 0)).copy()
                else:
                    broadcasted_transpose = broadcasted_tensor.copy()

                graph.get_buffer(fc_node.input_names[1]).producer.op.tensor = broadcasted_transpose
                graph.get_buffer(fc_node.input_names[2]).producer.op.tensor = bias * bn_node_weights + bn_node_bias
                graph.squash(bn_node, input_name=bn_input_buffer.name)
                log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                        fc_node.op.type,
                                                                                        fc_node.op.name))
                # Transferring activation encoding of BN to fullyconnected.
                q = graph.user_quantization_overrides
                if q and 'activation_encodings' in q and bn_output_buffer.name in q['activation_encodings']:
                    activations = q['activation_encodings']
                    act_encs = [IROptimizations.extract_encoding_dict(fc_node_output_buffer.name, activations[bn_output_buffer.name])]
                    graph.add_quantization_params(fc_node.op.name, output_encodings=act_encs)

    def prepare_inputs_as_params(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        bias_buffer = graph.get_buffer(node.input_names[2])
        bias_node = bias_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
            node.op.bias = bias_node.op.tensor
            # Remove the weights/bias inputs from the IR graph
            graph.remove_node_as_consumer(node, weights_buffer.name)
            graph.remove_node_as_consumer(node, bias_buffer.name)
            node.input_names = [node.input_names[0]]

    @staticmethod
    def squash_sum(graph):
        def validate_dims(nodes_tuple):
            fc_node = nodes_tuple[0]
            post_reshape_node = nodes_tuple[1]
            elementwise_sum_node = nodes_tuple[2]
            fc_bias_buffer = graph.get_input_buffers(fc_node)[2]
            elementwise_sum_input_buffers = graph.get_input_buffers(elementwise_sum_node)
            elementwise_sum_constant_buffer = elementwise_sum_input_buffers[0]
            if elementwise_sum_input_buffers[0].producer.op.TRANSLATION_KEY != 'constant':
                elementwise_sum_constant_buffer = elementwise_sum_input_buffers[1]
            fc_input_buffer_0 = graph.get_input_buffers(fc_node)[0]
            fc_input_buffer_1 = graph.get_input_buffers(fc_node)[1]
            q = graph.user_quantization_overrides
            # For simplified logic, currently we only support this merge when the bias input to FC is zeros.
            if not np.all(fc_bias_buffer.producer.op.tensor == 0):
                return False
            if q and 'activation_encodings' in q and elementwise_sum_constant_buffer.name in q['activation_encodings']:
                encoding = IROptimizations.extract_encoding_dict(elementwise_sum_constant_buffer.name, q['activation_encodings'][elementwise_sum_constant_buffer.name])
                # After this optimization, the constant input to the elementwise sum will be the bias for the FC.
                # External overrides (for the constant input) can have bitwidth that is not equal to 8 or 32.
                # Since the bitwidth for the bias can only be 8 or 32 bits, we cannot do this optimization.
                if encoding['bw'] != 8 and encoding['bw'] != 32:
                    return False
            if fc_bias_buffer.shape == elementwise_sum_constant_buffer.shape \
                    and fc_node.output_names[0] in post_reshape_node.input_names \
                    and post_reshape_node.output_names[0] in elementwise_sum_node.input_names \
                    and fc_input_buffer_0.rank() == 2 \
                    and fc_input_buffer_1.rank() == 2 \
                    and fc_bias_buffer.rank() == 1:
                return True
            return False

        sequence = [
            ("FullyConnected",
             ("MATCH_BUFS_AT_INDEX", [("Reshape", 0), ("constant", 1), ("constant", 2)]), ("MATCH_NUM_BUFS", [("Reshape", "ANY")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("FullyConnected", "ANY")]), ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("constant", "ANY"), ("Reshape", "ANY")]), ()
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_dims, ignore_constants=True)
        for node_tuple in matched_node_list:
            fc_node = node_tuple[0]
            reshape_node = node_tuple[1]
            elementwise_sum_node = node_tuple[2]
            fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
            fc_bias_node = graph.get_op_input_nodes(fc_node)[2]
            elementwise_sum_input_buffers = graph.get_input_buffers(elementwise_sum_node)
            elementwise_sum_constant_buffer = elementwise_sum_input_buffers[0]
            if elementwise_sum_input_buffers[0].producer.op.TRANSLATION_KEY != 'constant':
                elementwise_sum_constant_buffer = elementwise_sum_input_buffers[1]
            elementwise_sum_constant_node = [node for node in graph.get_op_input_nodes(elementwise_sum_node) if node.op.TRANSLATION_KEY == 'constant'][0]
            elementwise_sum_output_buffer = graph.get_output_buffers(elementwise_sum_node)[0]
            fc_bias_buffer = graph.get_output_buffers(fc_bias_node)[0]

            # Replacing Bias node with elementwise sum constant input node
            fc_bias_buffer.consumers.clear()
            fc_node.input_names[2] = elementwise_sum_constant_buffer.name
            elementwise_sum_constant_buffer.consumers.add(fc_node)
            graph.remove_node_as_consumer(elementwise_sum_node, elementwise_sum_constant_buffer.name)
            graph.squash(elementwise_sum_node, graph.get_output_buffers(reshape_node)[0].name)

            # update order from [fully_connected, reshape, bias] to [bias, fully_connected, reshape]
            idx_bias = graph.nodes_in_order.index(elementwise_sum_constant_node)
            idx_fc = graph.nodes_in_order.index(fc_node)
            if idx_bias > idx_fc:
                graph.nodes_in_order.pop(idx_bias)
                graph.nodes_in_order.insert(idx_fc, elementwise_sum_constant_node)

            # Transferring activation encoding of elementwise sum to fullyconnected.
            q = graph.user_quantization_overrides
            if q and 'activation_encodings' in q and elementwise_sum_output_buffer.name in q['activation_encodings']:
                activations = q['activation_encodings']
                act_encs = [IROptimizations.extract_encoding_dict(fc_node_output_buffer.name, activations[elementwise_sum_output_buffer.name])]
                graph.add_quantization_params(fc_node.op.name, output_encodings=act_encs)

@register_layer_optimization
class OptimizeGatherTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherOp.TRANSLATION_KEY
        self.register_method(INJECT_CAST_FOR_GATHER, self.inject_cast_for_gather)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)
        self.register_method(HANDLE_GATHER_NEGATIVE_INDICES, self.handle_gather_negative_indices)

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis and if needed permute it for NSC
        # In addition, output buffer axis tracking stays the same as input so long
        # as the rank of indices == 1. Otherwise it's non trivial as the rank will change
        input_name = node.input_names[0]
        indices_name = node.input_names[1]
        input_buf = graph.get_input_buffers(node)[0]
        indices_buf = graph.get_input_buffers(node)[1]
        output_buf = graph.get_output_buffers(node)[0]
        if node.op.axis < 0:
            node.op.axis = node.op.axis+input_buf.rank()
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            if indices_buf.rank() > 1:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                axis_map = graph.src_axis_order.permute_sequence_from_ir[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]
                output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            if indices_buf.rank() > 1:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                axis_map = graph.src_axis_order.permute_sequence_from_ir[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]
                output_buf.axis_format = AxisTracker.AxisFormat.NSC
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            if indices_buf.rank() > 1:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                axis_map = graph.src_axis_order.permute_sequence_from_ir[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]
                output_buf.axis_format = AxisTracker.AxisFormat.NFC
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCF_TO_NFC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            axis = node.op.axis
            indices_shape = indices_buf.shape
            output_shape = output_buf.shape
            indices_rank = len(indices_shape)
            # insert permute if indices shape doesn't match the shape of output
            if indices_shape != output_shape[axis:axis+indices_rank]:
                if indices_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                        indices_buf.axis_format != node.op.data_axis_formats[1]:
                    graph.inject_implicit_permute(indices_name, AxisTracker.AxisFormat.NCF,
                                                  AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
                elif indices_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                        indices_buf.axis_format != node.op.data_axis_formats[1]:
                    graph.inject_implicit_permute(indices_name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                elif indices_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                        indices_buf.axis_format != node.op.data_axis_formats[1]:
                    graph.inject_implicit_permute(indices_name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            if indices_buf.rank() > 1:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                output_buf.axis_format = input_buf.axis_format

        return True

    def handle_gather_negative_indices(self, node, graph):
        indices_name = node.input_names[1]
        if isinstance(graph.get_producer_op(indices_name), op_adapter.ConstantOp):
            const_op = graph.get_producer_op(indices_name)
            input_data_shape = graph.get_buffer(node.input_names[0]).shape
            with np.nditer(const_op.tensor, op_flags=['readwrite']) as it:
                for index in it:
                    if index < 0:
                        index += input_data_shape[node.op.axis]

    # TODO Remove this optimization once casts are properly optimized out in IR
    def inject_cast_for_gather(self, node, graph):
        cast_node_name = node.input_names[1] + "_cast"
        cast_op = op_adapter.CastOp(name=cast_node_name, to_type="int32")
        # check and reuse existing CastOp if already added
        if graph.has_buffer(cast_node_name):
            cast_buffer = graph.buffers[cast_node_name]
            cast_buffer.consumers.add(node)
            input_buffer = graph.buffers[node.input_names[1]]
            input_buffer.consumers.remove(node)
            node.input_names[1] = cast_node_name
        else:
            log_debug("Injecting cast op {} for node {}'s indices input.".format(cast_node_name, node.op.name))
            graph.inject(cast_op, input_name=node.input_names[1], output_name=cast_node_name, consumer_names=[node.op.name])

    @staticmethod
    def remove_identity(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        indices_buffer = graph.get_input_buffers(node)[1]
        output_buffer_shape = graph.get_output_buffers(node)[0].shape
        if input_buffer.shape == output_buffer_shape and len(input_buffer.consumers) == 1 and \
                indices_buffer.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # examine cases that indices make output exactly the same as input
            # i.e. indices = [0,1,...,n-1] along the gather axis w/ dim=n
            if all(indices_buffer.producer.op.tensor == list(range(input_buffer.shape[node.op.axis]))):
                # remove current gather op from indices op's consumers
                if node in indices_buffer.consumers:
                    indices_buffer.consumers.remove(node)
                    node.input_names.remove(indices_buffer.name)
                # this gather has no effect, remove indices first
                if len(indices_buffer.consumers) == 0:
                    indices_node = indices_buffer.producer
                    graph.prune(indices_node, force_remove=True)
                # then remove gather
                ret = graph.squash(node, input_name=input_buffer.name)
                if ret:
                    log_debug("Squash Gather op {} due to IdentityOp. "
                              "Input shape {}".format(node.op.name,
                                                      input_buffer.shape))

@register_layer_optimization
class OptimizeGatherElementsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherElementsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis
        input_name, indices_name = node.input_names
        input_buf, indices_buf = graph.get_input_buffers(node)
        if node.op.axis < 0:
            node.op.axis = node.op.axis+input_buf.rank()

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                log_debug1("{} axis is already in spatial first order {}, no need to reorder.", buf_name, buf_axis_format)
                return
            elif buf_axis_format in [AxisTracker.AxisFormat.NDHWC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NSC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NFC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NTF,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])

        return True

@register_layer_optimization
class OptimizeGatherNDTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherNDOp.TRANSLATION_KEY
        self.register_method(MATCH_GATHERND, self.match_gathernd)

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name = node.input_names
        input_buf, indices_buf = graph.get_input_buffers(node)

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                log_debug1("{} axis is already in spatial first order {}, no need to reorder.", buf_name, buf_axis_format)
                return
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if AxisTracker.input_axis_formats_intact(graph, node):
            return False

        # Check and reorder buffers axis to spatial first.
        # All inputs need to be in source framework order.
        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])

        return True

    @staticmethod
    def match_gathernd(graph):
        sequence = [
            (ir_graph.QNN_OP_TRANSPOSE,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_TRANSPOSE, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            (ir_graph.QNN_OP_GATHER,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ANY"), ("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_TRANSPOSE, "ALL")])
            ),
            (ir_graph.QNN_OP_TRANSPOSE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_TRANSPOSE, "ALL")]),
             ()
            )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)
        for node_tuple in matched_node_list:
            node_types_seq = [node.op.type for node in node_tuple]
            gather_node_idx = node_types_seq.index(ir_graph.QNN_OP_GATHER)
            gather_node = node_tuple[gather_node_idx]

            # Turn Add(Mul(X, c)+Y) into Pack([X, Y], axis=2)
            multi_node = node_tuple[2]
            graph.squash(multi_node, multi_node.input_names[0])

            add_node = node_tuple[3]
            add_node_consumers = graph.get_output_buffers(add_node)[0].consumers

            pack_op = op_adapter.PackOp(add_node.op.name + '_pack', axis=2)
            graph.replace(add_node.op, pack_op)

            # Keep first Transpose and Gather, then squash the rest of nodes in sequence
            graph.squash(node_tuple[1], node_tuple[1].input_names[0], squash_into_next=True)
            for node in node_tuple[1:]:
                if node != gather_node and node in graph.list_nodes():
                    graph.squash(node, node.input_names[0])

            # Replace Gather by GatherND
            data_shape = graph.get_input_shapes(gather_node)[0]
            indices_shape = graph.get_input_shapes(gather_node)[1]

            gathernd_op = op_adapter.GatherNDOp(name=gather_node.op.name, batch_dims=0)
            graph.replace(gather_node.op, gathernd_op)

            permute_op_name = gather_node.op.name + '_permute'
            permute_op = op_adapter.TransposeOp(permute_op_name, perm=[2,3,0,1])
            permute_node = graph.inject(permute_op, input_name=gather_node.output_names[0], output_name=permute_op_name)


@register_layer_optimization
class OptimizeGeluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GeluOp.TRANSLATION_KEY
        self.register_method(MATCH_GELU, self.match_gelu)

    @staticmethod
    def match_gelu(graph):
        sequence1 = [
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Erf", "ALL")])
             ),
            ("Erf",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]
        sequence2 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Erf", "ALL")])
             ),
            ("Erf",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]
        sequence3 = [
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Erf", "ALL")])
             ),
            ("Erf",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum", "ANY")]),
             ()
             )
        ]
        sequences = [sequence1, sequence2, sequence3]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)
            for node_tuple in matched_node_list:
                last_node = node_tuple[-1]
                # Squash all nodes except the last node in forward order and the last op will be replaced
                for node in node_tuple[:-1]:
                    input_names = node.input_names[:]
                    # pick squashable input based on whether it produced by constantOp
                    input_name = [name for name in input_names if (
                                not isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                    input_names.remove(input_name)
                    for input_name_ in input_names:
                        # disconnect rest of inputs from node
                        # skip if current input_name_ equal to the input_name to squash
                        if input_name_ == input_name:
                            continue
                        input_buf_ = graph.get_buffer(input_name_)
                        input_buf_.consumers.remove(node)
                        node.input_names.remove(input_name_)
                    graph.squash(node, input_name=input_name, squash_into_next=True)

                # For the last_node, three different sequences correspond to two different processes:
                # Sequence1:
                # the inputs of last elementwise_product op will be [original input, constant(0.5)].
                # so we need to disconnect the constant input of the last op
                # Sequence2, sequence3:
                # the last elementwise_product op will receive two duplicated input buffer
                # after squashing previous nodes, so we need pop one of them.
                const_input_bufs = [graph.get_buffer(name) for name in last_node.input_names if
                                graph.get_producer_op(name).type == op_adapter.ConstantOp.TRANSLATION_KEY]
                if len(const_input_bufs):
                    const_input_bufs[0].consumers.remove(last_node)
                    last_node.input_names.remove(const_input_bufs[0].name)
                else:
                    last_node.input_names.pop()

                # replace the first op with gelu
                last_node_op = last_node.op
                gelu_op_name =  graph.naming_policy.get_op_name_by_type(op_adapter.GeluOp.type,
                                                                        op_adapter.GeluOp.LEGACY_TRANSLATION_KEY)
                gelu_op = op_adapter.GeluOp(gelu_op_name)
                graph.replace(last_node_op, gelu_op)


@register_layer_optimization
class OptimizeGenerateProposalsOp(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GenerateProposalsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeGridSampleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GridSampleOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        log_assert(len(input_bufs[0].shape) in [4, 5],
                   "GridSample op only support 4D or 5D input shape, got input shape {} for op {}.".format(input_bufs[0].shape, node.op.name))

        if input_bufs[0].axis_format == node.op.data_axis_formats[0] and \
                input_bufs[0].axis_format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC]:
            # No change
            return False
        elif input_bufs[0].axis_format == AxisTracker.AxisFormat.NCS:
            graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NSC,
                                          AxisTracker.AxisFormat.NCS_TO_NSC, [node.op.name])
        elif input_bufs[0].axis_format == AxisTracker.AxisFormat.NCDHW:
            graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NDHWC,
                                          AxisTracker.AxisFormat.NCDHW_TO_NDHWC, [node.op.name])
        elif input_bufs[0].axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            if len(input_bufs[0].shape) == 4:
                graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NSC,
                                              AxisTracker.AxisFormat.NCS_TO_NSC, [node.op.name])
            elif len(input_bufs[0].shape) == 5:
                graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NDHWC,
                                              AxisTracker.AxisFormat.NCDHW_TO_NDHWC, [node.op.name])

        # The second input is NONTRIVIAL, and we don't want to change the shape order,
        # so set enforce_input_spatial_last to False here. Otherwise, it will be set to True by default
        AxisTracker.image_to_channel_last_order(node, graph, enforce_input_spatial_last=False)
        return True


@register_layer_optimization
class OptimizeGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GruOp.TRANSLATION_KEY
        self.register_method(UNROLL_GRU_TIME_STEPS, self.unroll_gru_time_steps)
        self.register_method(EXPAND_GRU_OP_STRUCTURE, self.expand_gru_op_structure)

    def axes_to_spatial_first_order(self, node, graph):
        # GRU input axis format must be NTF
        input_name = node.input_names[0]
        input_bufs = graph.get_input_buffers(node)
        output_bufs = graph.get_output_buffers(node)

        # Enforce Data Buffer is NTF
        in_buf = input_bufs[0]
        if in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                in_buf.axis_format == AxisTracker.AxisFormat.TNF:
            graph.inject_implicit_permute(in_buf.name, AxisTracker.AxisFormat.NTF,
                                          AxisTracker.AxisFormat.TNF_TO_NTF)

        # Check that h input buffer is NONTRIVIAL
        for i, in_buf in enumerate(input_bufs[1:]):
            log_assert(in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL,
                       "GRU h input buffer {} needs to have format NONTRIVIAL, got {}",
                       in_buf,
                       in_buf.axis_format)

        # Set up GRU outputs' axis formats
        # First output: NTF
        # Other outputs: NONTRIVIAL
        for i, output_buf in enumerate(output_bufs):
            if i == 0:
                output_buf.axis_format = AxisTracker.AxisFormat.NTF
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    def unroll_gru_time_steps(self, graph):
        sequence = [
            (op_adapter.GruOp.TRANSLATION_KEY, (), ())
        ]

        def validate_node(nodes_tuple):
            batch_size, seq_length, input_size = graph.get_buffer(nodes_tuple[0].input_names[0]).shape[:]
            return seq_length > 1

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for nodes_tuple in matched_node_list:
            gru_node = nodes_tuple[0]
            gru_node_name = gru_node.op.name

            DATA_IDX, HIDDEN_OUT_IDX = 0, 1
            gru_node_input_name = gru_node.input_names[DATA_IDX]
            gru_node_output_name = gru_node.output_names[DATA_IDX]
            gru_node_idx = graph.nodes_in_order.index(gru_node)

            log_debug("Unrolling GRU node {}".format(gru_node_name))

            # Extract and validate inputs, outputs, and sizes
            number_of_outputs = len(gru_node.output_names)
            all_output_buffer = graph.get_buffer(gru_node_output_name)
            batch_size, seq_length, input_size = graph.get_buffer(gru_node_input_name).shape[:]

            if number_of_outputs == 1:
                # Add dummy buffers for missing outputs
                output_size = graph.get_buffer(gru_node_output_name).shape[-1]
                num_units = gru_node.op.hidden_size
                hidden_output_dummy_name = gru_node_name + "_hidden_output_dummy"
                graph.add_output_buffer(gru_node, hidden_output_dummy_name,
                                        [batch_size, output_size], AxisTracker.AxisFormat.NONTRIVIAL)

            hidden_output_buffer = graph.get_buffer(gru_node.output_names[HIDDEN_OUT_IDX])

            input_x_split_name_list = []
            for i in range(seq_length):
                input_x_i_name = gru_node_name + "_" + gru_node_input_name + str(i)
                input_x_split_name_list.append(input_x_i_name)
            input_x_split_name = gru_node_name + "_" + gru_node_input_name + "_split"
            time_step_axis = 1
            input_x_split_op = op_adapter.SplitOp(name=input_x_split_name, axis=time_step_axis)

            # Split input to T inputs
            graph.add(input_x_split_op, input_names=[gru_node.input_names[0]],
                                        output_names=input_x_split_name_list, idx=graph.nodes_in_order.index(gru_node))
            if gru_node.op.backward:
                input_x_split_name_list.reverse()

            output_y_concat_name_list = []
            output_h_name_list = []
            for i in range(seq_length):
                output_y_i_name = gru_node_output_name + str(i)
                output_y_concat_name_list.append(output_y_i_name)
                output_h_i_name = gru_node.output_names[HIDDEN_OUT_IDX] + str(i)
                output_h_name_list.append(output_h_i_name)

            for i in range(seq_length):
                if i == 0:
                    #_reset_state_at_time_step_0 = gru_node.op.reset_state_at_time_step_0
                    _h_0_input_name = gru_node.input_names[HIDDEN_OUT_IDX]
                else:
                    #_reset_state_at_time_step_0 = False
                    _h_0_input_name = output_h_name_list[i-1]

                gru_time_step_i_op_name = gru_node_name + str(i)
                gru_time_step_i_op = op_adapter.GruOp(name=gru_time_step_i_op_name,
                                                      state_gate=gru_node.op.state_gate,
                                                      forget_gate=gru_node.op.forget_gate,
                                                      control_gate=gru_node.op.control_gate,
                                                      activation=gru_node.op.activation,
                                                      gate_activation=gru_node.op.gate_activation,
                                                      rec_gate_activation=gru_node.op.rec_gate_activation,
                                                      h_0_input_name=_h_0_input_name,
                                                      backward=gru_node.op.backward,
                                                      hidden_size=gru_node.op.hidden_size,
                                                      linear_before_reset=gru_node.op.linear_before_reset)

                gru_op_input_name_list = [input_x_split_name_list[i], _h_0_input_name]
                gru_op_output_name_list = [output_y_concat_name_list[i], output_h_name_list[i]]
                graph.add(gru_time_step_i_op, input_names=gru_op_input_name_list, output_names=gru_op_output_name_list, idx=graph.nodes_in_order.index(gru_node))

            output_y_concat_name = gru_node_output_name + "_concat"
            output_y_concat_op_name = gru_node_name + "_" + gru_node_output_name + "_concat"
            output_y_concat_op = op_adapter.ConcatOp(name=output_y_concat_op_name, axis=time_step_axis)

            # Concat output from T outputs
            graph.add(output_y_concat_op, input_names=output_y_concat_name_list, output_names=output_y_concat_name, idx=graph.nodes_in_order.index(gru_node))

            for consumer in list(all_output_buffer.consumers):
                output_y_concat_buffer = graph.get_buffer(output_y_concat_name)
                output_y_concat_buffer.consumers.add(consumer)
                consumer.input_names.append(output_y_concat_name)
            for consumer in list(hidden_output_buffer.consumers):
                output_h_buffer = graph.get_buffer(output_h_name_list[seq_length-1])
                output_h_buffer.consumers.add(consumer)
                consumer.input_names.append(output_h_name_list[seq_length-1])

            # prune original gru_node
            graph.prune(gru_node, force_remove=True)

    # (1) - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    def expand_gru_update_gate(self, graph, gru_node, input_x_reshape_node_name, input_h_reshape_node_name):
        gru_node_name = gru_node.op.name
        gru_node_idx = graph.nodes_in_order.index(gru_node)

        # update gate (z)
        Wz = gru_node.op.control_gate["weights"]
        Rz = gru_node.op.control_gate["rec_weights"]
        WBz = gru_node.op.control_gate["fwd_bias"]
        RBz = gru_node.op.control_gate["rec_bias"]
        Bz = np.add(WBz, RBz)

        input_to_control_w_name = gru_node_name + '_input_to_control_w'
        input_to_control_w_op = op_adapter.ConstantOp(name=input_to_control_w_name, tensor=Wz)
        graph.add(input_to_control_w_op, input_names=[], output_names=[input_to_control_w_name], idx=gru_node_idx-1)

        recurrent_to_control_w_name = gru_node_name + '_recurrent_to_control_w'
        recurrent_to_control_w_op = op_adapter.ConstantOp(name=recurrent_to_control_w_name, tensor=Rz)
        graph.add(recurrent_to_control_w_op, input_names=[], output_names=[recurrent_to_control_w_name], idx=gru_node_idx-1)

        control_gate_b_name = gru_node_name + '_control_gate_b'
        control_gate_b_op = op_adapter.ConstantOp(name=control_gate_b_name, tensor=Bz)
        graph.add(control_gate_b_op, input_names=[], output_names=[control_gate_b_name], idx=gru_node_idx-1)

        xt_wz_wbz_matmul_bias_op_name = gru_node_name + "_xt_wz_wbz_matmul_bias_op"
        Wbz = np.zeros(graph.get_buffer(control_gate_b_name).shape[-1], dtype=np.float32)
        xt_wz_wbz_matmul_bias_op = op_adapter.MatMulOp(name=xt_wz_wbz_matmul_bias_op_name,
                                                        bias=Wbz,
                                                        transpose_a=False,
                                                        transpose_b=False)
        graph.add(xt_wz_wbz_matmul_bias_op, input_names=[input_x_reshape_node_name, input_to_control_w_name],
                            output_names=[xt_wz_wbz_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        ht_1_rz_rbz_matmul_bias_op_name = gru_node_name + "_ht_1_rz_rbz_matmul_bias_op"
        Rbz = np.zeros(graph.get_buffer(control_gate_b_name).shape[-1], dtype=np.float32)
        ht_1_rz_rbz_matmul_bias_op = op_adapter.MatMulOp(name=ht_1_rz_rbz_matmul_bias_op_name,
                                                            bias=Rbz,
                                                            transpose_a=False,
                                                            transpose_b=False)
        graph.add(ht_1_rz_rbz_matmul_bias_op, input_names=[input_h_reshape_node_name, recurrent_to_control_w_name],
                            output_names=[ht_1_rz_rbz_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        elementsum_of_term1_op_name = gru_node_name + "_elementsum_of_term1_op"
        elementsum_of_term1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term1_op, input_names=[xt_wz_wbz_matmul_bias_op_name, ht_1_rz_rbz_matmul_bias_op_name],
                            output_names=[elementsum_of_term1_op_name], idx=graph.nodes_in_order.index(gru_node))

        elementsum_of_term1_bias_op_name = gru_node_name + "_elementsum_of_term1_bias_op"
        elementsum_of_term1_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term1_bias_op, input_names=[elementsum_of_term1_op_name, control_gate_b_name],
                            output_names=[elementsum_of_term1_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        activation_of_term1_op_name = gru_node_name + "_activation_term1_op"
        activation_of_term1_op = op_adapter.NeuronOp(name=activation_of_term1_op_name, neuron_type=gru_node.op.activation)
        graph.add(activation_of_term1_op, input_names=[elementsum_of_term1_bias_op_name],
                            output_names=[activation_of_term1_op_name], idx=graph.nodes_in_order.index(gru_node))

        return activation_of_term1_op_name

    # (2) - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    def expand_gru_reset_gate(self, graph, gru_node, input_x_reshape_node_name, input_h_reshape_node_name):
        gru_node_name = gru_node.op.name
        gru_node_idx = graph.nodes_in_order.index(gru_node)

        # reset gate (r)
        Wr = gru_node.op.forget_gate["weights"]
        Rr = gru_node.op.forget_gate["rec_weights"]
        WBr = gru_node.op.forget_gate['fwd_bias']
        RBr = gru_node.op.forget_gate['rec_bias']
        Br = np.add(WBr, RBr)

        input_to_forget_w_name = gru_node_name + '_input_to_forget_w'
        input_to_forget_w_op = op_adapter.ConstantOp(name=input_to_forget_w_name, tensor=Wr)
        graph.add(input_to_forget_w_op, input_names=[], output_names=[input_to_forget_w_name], idx=gru_node_idx-1)

        recurrent_to_forget_w_name = gru_node_name + '_recurrent_to_forget_w'
        recurrent_to_forget_w_op = op_adapter.ConstantOp(name=recurrent_to_forget_w_name, tensor=Rr)
        graph.add(recurrent_to_forget_w_op, input_names=[], output_names=[recurrent_to_forget_w_name], idx=gru_node_idx-1)

        forget_gate_b_name = gru_node_name + '_forget_gate_b'
        forget_gate_b_op = op_adapter.ConstantOp(name=forget_gate_b_name, tensor=Br)
        graph.add(forget_gate_b_op, input_names=[], output_names=[forget_gate_b_name], idx=gru_node_idx-1)

        xt_wr_wbr_matmul_bias_op_name = gru_node_name + "_xt_wr_wbr_matmul_bias_op"
        Wbr = np.zeros(graph.get_buffer(forget_gate_b_name).shape[-1], dtype=np.float32)
        xt_wr_wbr_matmul_bias_op = op_adapter.MatMulOp(name=xt_wr_wbr_matmul_bias_op_name,
                                                        bias=Wbr,
                                                        transpose_a=False,
                                                        transpose_b=False)
        graph.add(xt_wr_wbr_matmul_bias_op, input_names=[input_x_reshape_node_name, input_to_forget_w_name],
                            output_names=[xt_wr_wbr_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        ht_1_rr_rbr_matmul_bias_op_name = gru_node_name + "_ht_1_rr_rbr_matmul_bias_op"
        Rbr = np.zeros(graph.get_buffer(forget_gate_b_name).shape[-1], dtype=np.float32)
        ht_1_rr_rbr_matmul_bias_op = op_adapter.MatMulOp(name=ht_1_rr_rbr_matmul_bias_op_name,
                                                            bias=Rbr,
                                                            transpose_a=False,
                                                            transpose_b=False)
        graph.add(ht_1_rr_rbr_matmul_bias_op, input_names=[input_h_reshape_node_name, recurrent_to_forget_w_name],
                            output_names=[ht_1_rr_rbr_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        elementsum_of_term2_op_name = gru_node_name + "_elementsum_of_term2_op"
        elementsum_of_term2_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term2_op, input_names=[xt_wr_wbr_matmul_bias_op_name, ht_1_rr_rbr_matmul_bias_op_name],
                            output_names=[elementsum_of_term2_op_name], idx=graph.nodes_in_order.index(gru_node))

        elementsum_of_term2_bias_op_name = gru_node_name + "_elementsum_of_term2_bias_op"
        elementsum_of_term2_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term2_bias_op, input_names=[elementsum_of_term2_op_name, forget_gate_b_name],
                            output_names=[elementsum_of_term2_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        activation_of_term2_op_name = gru_node_name + "_activation_term2_op"
        activation_of_term2_op = op_adapter.NeuronOp(name=activation_of_term2_op_name, neuron_type=gru_node.op.gate_activation)
        graph.add(activation_of_term2_op, input_names=[elementsum_of_term2_bias_op_name],
                            output_names=[activation_of_term2_op_name], idx=graph.nodes_in_order.index(gru_node))

        return activation_of_term2_op_name

    # (3.1) - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) if linear_before_reset == 0 (default)
    # (3.2) - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) if linear_before_reset != 0
    def expand_gru_hidden_gate(self, graph, gru_node, input_x_reshape_node_name, input_h_reshape_node_name, rt, linear_before_reset = 0):
        gru_node_name = gru_node.op.name
        gru_node_idx = graph.nodes_in_order.index(gru_node)

        # hidden gate(h)
        Wh = gru_node.op.state_gate["weights"]
        Rh = gru_node.op.state_gate["rec_weights"]
        WBh = gru_node.op.state_gate["fwd_bias"]
        RBh = gru_node.op.state_gate["rec_bias"]
        Bh = np.add(WBh, RBh)

        input_to_state_w_name = gru_node_name + '_input_to_state_w'
        input_to_state_w_op = op_adapter.ConstantOp(name=input_to_state_w_name, tensor=Wh)
        graph.add(input_to_state_w_op, input_names=[], output_names=[input_to_state_w_name], idx=gru_node_idx-1)

        recurrent_to_state_w_name = gru_node_name + '_recurrent_to_state_w'
        recurrent_to_state_w_op = op_adapter.ConstantOp(name=recurrent_to_state_w_name, tensor=Rh)
        graph.add(recurrent_to_state_w_op, input_names=[], output_names=[recurrent_to_state_w_name], idx=gru_node_idx-1)

        state_gate_fwd_b_name = gru_node_name + '_state_gate_fwd_b'
        state_gate_fwd_b_op = op_adapter.ConstantOp(name=state_gate_fwd_b_name, tensor=WBh)
        graph.add(state_gate_fwd_b_op, input_names=[], output_names=[state_gate_fwd_b_name], idx=gru_node_idx-1)

        state_gate_rec_b_name = gru_node_name + '_state_gate_rec_b'
        state_gate_rec_b_op = op_adapter.ConstantOp(name=state_gate_rec_b_name, tensor=RBh)
        graph.add(state_gate_rec_b_op, input_names=[], output_names=[state_gate_rec_b_name], idx=gru_node_idx-1)

        xt_wh_wbh_matmul_op_name = gru_node_name + "_xt_wh_wbh_matmul_op"
        Wbh = np.zeros(graph.get_buffer(state_gate_fwd_b_name).shape[-1], dtype=np.float32)
        xt_wh_wbh_matmul_bias_op = op_adapter.MatMulOp(name=xt_wh_wbh_matmul_op_name,
                                                        bias=Wbh,
                                                        transpose_a=False,
                                                        transpose_b=False)
        graph.add(xt_wh_wbh_matmul_bias_op, input_names=[input_x_reshape_node_name, input_to_state_w_name],
                            output_names=[xt_wh_wbh_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))

        xt_wh_wbh_matmul_bias_op_name = gru_node_name + "_xt_wh_wbh_matmul_bias_op"
        xt_wh_wbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=xt_wh_wbh_matmul_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(xt_wh_wbh_matmul_bias_op, input_names=[xt_wh_wbh_matmul_op_name, state_gate_fwd_b_name],
                            output_names=[xt_wh_wbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

        activation_of_term3_op_name = ''
        if linear_before_reset == 0:
            rt_dot_ht_1_op_name = gru_node_name + "_rt_dot_ht_1_op"
            rt_dot_ht_1_op = op_adapter.ElementwiseBinaryOp(name=rt_dot_ht_1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
            graph.add(rt_dot_ht_1_op, input_names=[rt, input_h_reshape_node_name],
                                output_names=[rt_dot_ht_1_op_name], idx=graph.nodes_in_order.index(gru_node))

            rt_dot_ht_1_rh_rbh_matmul_op_name = gru_node_name + "_rt_dot_ht_1_rh_rbh_matmul_op"
            Rbh = np.zeros(graph.get_buffer(state_gate_rec_b_name).shape[-1], dtype=np.float32)
            rt_dot_ht_1_rh_rbh_matmul_op = op_adapter.MatMulOp(name=rt_dot_ht_1_rh_rbh_matmul_op_name,
                                                                bias=Rbh,
                                                                transpose_a=False,
                                                                transpose_b=False)
            graph.add(rt_dot_ht_1_rh_rbh_matmul_op, input_names=[rt_dot_ht_1_op_name, recurrent_to_state_w_name],
                                output_names=[rt_dot_ht_1_rh_rbh_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))

            rt_dot_ht_1_rh_rbh_matmul_bias_op_name = gru_node_name + "_rt_dot_ht_1_rh_rbh_matmul_bias_op"
            rt_dot_ht_1_rh_rbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=rt_dot_ht_1_rh_rbh_matmul_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(rt_dot_ht_1_rh_rbh_matmul_bias_op, input_names=[rt_dot_ht_1_rh_rbh_matmul_op_name, state_gate_rec_b_name],
                                output_names=[rt_dot_ht_1_rh_rbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

            elementsum_of_term3p1_op_name = gru_node_name + "_elementsum_of_term3p1_op"
            elementsum_of_term3p1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term3p1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(elementsum_of_term3p1_op, input_names=[xt_wh_wbh_matmul_bias_op_name, rt_dot_ht_1_rh_rbh_matmul_bias_op_name],
                                output_names=[elementsum_of_term3p1_op_name], idx=graph.nodes_in_order.index(gru_node))

            activation_of_term3_op_name = gru_node_name + "_activation_term3p1_op"
            activation_of_term3p1_op = op_adapter.NeuronOp(name=activation_of_term3_op_name, neuron_type=gru_node.op.rec_gate_activation)
            graph.add(activation_of_term3p1_op, input_names=[elementsum_of_term3p1_op_name],
                                output_names=[activation_of_term3_op_name], idx=graph.nodes_in_order.index(gru_node))
        else:
            ht_1_rh_rbh_matmul_op_name = gru_node_name + "ht_1_rh_rbh_matmul_op"
            Rbh = np.zeros(graph.get_buffer(state_gate_rec_b_name).shape[-1], dtype=np.float32)
            ht_1_rh_rbh_matmul_op = op_adapter.MatMulOp(name=ht_1_rh_rbh_matmul_op_name,
                                                        bias=Rbh,
                                                        transpose_a=False,
                                                        transpose_b=False)
            graph.add(ht_1_rh_rbh_matmul_op, input_names=[input_h_reshape_node_name, recurrent_to_state_w_name],
                                output_names=[ht_1_rh_rbh_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))

            ht_1_rh_rbh_matmul_bias_op_name = gru_node_name + "ht_1_rh_rbh_matmul_bias_op"
            ht_1_rh_rbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=ht_1_rh_rbh_matmul_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(ht_1_rh_rbh_matmul_bias_op, input_names=[ht_1_rh_rbh_matmul_op_name, state_gate_rec_b_name],
                                output_names=[ht_1_rh_rbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

            rt_dot_ht_1_rh_rbh_matmul_bias_op_name = gru_node_name + "_rt_dot_ht_1_rh_rbh_matmul_bias_op"
            rt_dot_ht_1_rh_rbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=rt_dot_ht_1_rh_rbh_matmul_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
            graph.add(rt_dot_ht_1_rh_rbh_matmul_bias_op, input_names=[rt, ht_1_rh_rbh_matmul_bias_op_name],
                                output_names=[rt_dot_ht_1_rh_rbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))

            elementsum_of_term3p2_op_name = gru_node_name + "_elementsum_of_term3p2_op"
            elementsum_of_term3p2_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term3p2_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(elementsum_of_term3p2_op, input_names=[xt_wh_wbh_matmul_bias_op_name, rt_dot_ht_1_rh_rbh_matmul_bias_op_name],
                                output_names=[elementsum_of_term3p2_op_name], idx=graph.nodes_in_order.index(gru_node))

            activation_of_term3_op_name = gru_node_name + "_activation_term3p2_op"
            activation_of_term3p2_op = op_adapter.NeuronOp(name=activation_of_term3_op_name, neuron_type=gru_node.op.rec_gate_activation)
            graph.add(activation_of_term3p2_op, input_names=[elementsum_of_term3p2_op_name],
                                output_names=[activation_of_term3_op_name], idx=graph.nodes_in_order.index(gru_node))

        return activation_of_term3_op_name

    # (4) - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    def update_gru_hidden_state(self, graph, gru_node, Ht_1, zt, ht):
        gru_node_name = gru_node.op.name
        gru_node_idx = graph.nodes_in_order.index(gru_node)

        initial_ones_op_name = gru_node_name + '_initial_ones_op'
        initial_ones_tensor = np.ones(graph.get_buffer(zt).shape, dtype=np.float32)
        initial_ones_op = op_adapter.ConstantOp(name=initial_ones_op_name, tensor=initial_ones_tensor)
        graph.add(initial_ones_op, input_names=[], output_names=[initial_ones_op_name], idx=gru_node_idx-1)

        one_minus_zt_op_name = gru_node_name + "_one_minus_zt_op"
        one_minus_zt_op = op_adapter.ElementwiseBinaryOp(name=one_minus_zt_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT)
        graph.add(one_minus_zt_op, input_names=[initial_ones_op_name, zt],
                            output_names=[one_minus_zt_op_name], idx=graph.nodes_in_order.index(gru_node))

        one_minus_zt_dot_ht_op_name = gru_node_name + '_one_minus_zt_dot_ht_op'
        one_minus_zt_dot_ht_op = op_adapter.ElementwiseBinaryOp(name=one_minus_zt_dot_ht_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
        graph.add(one_minus_zt_dot_ht_op, input_names=[one_minus_zt_op_name, ht],
                            output_names=[one_minus_zt_dot_ht_op_name], idx=graph.nodes_in_order.index(gru_node))

        zt_dot_ht_1_op_name = gru_node_name + '_zt_dot_ht_1_op'
        zt_dot_ht_1_op = op_adapter.ElementwiseBinaryOp(name=zt_dot_ht_1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
        graph.add(zt_dot_ht_1_op, input_names=[zt, Ht_1],
                            output_names=[zt_dot_ht_1_op_name], idx=graph.nodes_in_order.index(gru_node))

        elementsum_of_term4_op_name = gru_node_name + "_elementsum_of_term4_op"
        elementsum_of_term4_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term4_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term4_op, input_names=[one_minus_zt_dot_ht_op_name, zt_dot_ht_1_op_name],
                            output_names=[elementsum_of_term4_op_name], idx=graph.nodes_in_order.index(gru_node))

        return elementsum_of_term4_op_name

    def expand_gru_op_structure(self, graph):
        sequence = [
            (op_adapter.GruOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence)
        for nodes_tuple in matched_node_list:
            gru_node = nodes_tuple[0]
            gru_node_name = gru_node.op.name
            gru_node_idx = graph.nodes_in_order.index(gru_node)
            number_of_inputs = len(gru_node.input_names)
            number_of_outputs = len(gru_node.output_names)

            batch_size, seq_length, input_size = graph.get_buffer(gru_node.input_names[0]).shape[:]
            output_size = graph.get_buffer(gru_node.output_names[0]).shape[-1]

            # Check that extracted sequence length is 1
            if seq_length != 1:
                raise ValueError('Unsupported sequence length for GRU node {}, expected 1, got {}.'.format(
                        gru_node_name, seq_length))

            # Requires initial_h inputs to be present
            # The following code adds zero valued tensors provided the conditions below are satisfied
            if number_of_inputs != 2:
                if gru_node.op.h_0_input_name or len(gru_node.input_names) != 1:
                    raise ValueError('Unsupported number of inputs for GRU node {}, expected 2 or 1 if no initial states, got {}.'.format(
                        gru_node_name, len(gru_node.input_names)))

                # add zeros for initial h inputs since there are needed for QNN
                initial_hidden_state_name = gru_node_name + '_initial_hidden_state'
                initial_hidden_state_tensor = np.zeros((batch_size, output_size), dtype=np.float32)
                initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor)
                graph.add(initial_hidden_state_op, input_names=[], output_names=[initial_hidden_state_name], idx=gru_node_idx-1)
                gru_node.op.h_0_input_name = initial_hidden_state_name

                gru_node.input_names.extend([initial_hidden_state_name])

            # Only 1 or 2 outputs are supported for this optimization
            if number_of_outputs != 1 and number_of_outputs != 2:
                raise ValueError("Unsupported number of outputs for GRU node {}, expected 1 or 2, got {}.".format(
                    gru_node_name, number_of_outputs))

            Xt = gru_node.input_names[0]
            if len(graph.get_buffer(gru_node.input_names[0]).shape) == 3:
                Xt = gru_node_name + "_" + gru_node.input_names[0] + "_reshape"
                input_x_reshape_output_shape = [batch_size, input_size]
                input_x_reshape_op = op_adapter.ReshapeOp(name=Xt,
                                                          shape=input_x_reshape_output_shape)
                graph.add(input_x_reshape_op, input_names=[gru_node.input_names[0]],
                             output_names=[Xt], idx=graph.nodes_in_order.index(gru_node))

            Ht_1 = gru_node.input_names[1]
            if len(graph.get_buffer(gru_node.input_names[1]).shape) == 3:
                Ht_1 = gru_node_name + "_" + gru_node.input_names[1] + "_reshape"
                input_h_reshape_output_shape = [batch_size, output_size]
                input_h_reshape_op = op_adapter.ReshapeOp(name=Ht_1,
                                                          shape=input_h_reshape_output_shape)
                graph.add(input_h_reshape_op, input_names=[gru_node.input_names[1]],
                             output_names=[Ht_1], idx=graph.nodes_in_order.index(gru_node))

            # expand gru op structure
            # zt = f(Xt*(Wz^T) + Ht_1*(Rz^T) + Wbz + Rbz) ........(1)
            # rt = f(Xt*(Wr^T) + Ht_1*(Rr^T) + Wbr + Rbr) ........(2)
            # ht = g(Xt*(Wh^T) + (rt (.) Ht_1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0 ........(3.1)
            # ht = g(Xt*(Wh^T) + (rt (.) (Ht_1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0 ........(3.2)
            # Ht = (1 - zt) (.) ht + zt (.) Ht_1 ........(4)

            zt = self.expand_gru_update_gate(graph, gru_node, Xt, Ht_1) # ........(1)
            rt = self.expand_gru_reset_gate(graph, gru_node, Xt, Ht_1) # ........(2)
            ht = self.expand_gru_hidden_gate(graph, gru_node, Xt, Ht_1, rt, gru_node.op.linear_before_reset) # ........(3)
            Ht = self.update_gru_hidden_state(graph, gru_node, Ht_1, zt, ht) # ........(4)

            output_all_hiddens_reshape_node_name = gru_node.output_names[0] + "_reshape"
            output_all_hiddens_output_shape = graph.get_buffer(gru_node.output_names[0]).shape
            output_all_hiddens_reshape_op = op_adapter.ReshapeOp(name=output_all_hiddens_reshape_node_name,
                                                                 shape=output_all_hiddens_output_shape)
            graph.add(output_all_hiddens_reshape_op, input_names=[Ht],
                             output_names=[output_all_hiddens_reshape_node_name], idx=graph.nodes_in_order.index(gru_node))

            output_hidden_reshape_node_name = gru_node.output_names[1] + "_reshape"
            output_hidden_output_shape = graph.get_buffer(gru_node.output_names[1]).shape
            output_all_hiddens_reshape_op = op_adapter.ReshapeOp(name=output_hidden_reshape_node_name,
                                                                 shape=output_hidden_output_shape)
            graph.add(output_all_hiddens_reshape_op, input_names=[Ht],
                             output_names=[output_hidden_reshape_node_name], idx=graph.nodes_in_order.index(gru_node))

            for consumer in list(graph.get_buffer(gru_node.output_names[0]).consumers):
                output_all_hiddens_buffer = graph.get_buffer(output_all_hiddens_reshape_node_name)
                output_all_hiddens_buffer.consumers.add(consumer)
                if gru_node.op.backward:
                    consumer.input_names.insert(0, output_all_hiddens_reshape_node_name)
                else:
                    consumer.input_names.append(output_all_hiddens_reshape_node_name)
            for consumer in list(graph.get_buffer(gru_node.output_names[1]).consumers):
                output_hidden_reshape_buffer = graph.get_buffer(output_hidden_reshape_node_name)
                output_hidden_reshape_buffer.consumers.add(consumer)
                consumer.input_names.append(output_hidden_reshape_node_name)

            # prune original gru_node
            graph.prune(gru_node, force_remove=True)


@register_layer_optimization
class OptimizeIdentityTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.IdentityOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format
        return True

    @staticmethod
    def remove_identity(node, graph):
        try:
            graph.squash_identity(node)
        except:
            # Replace IdentityOp with ReshapeOp to avoid implementation issue when it can't be squashed
            input_buf = graph.get_input_buffers(node)[0]
            input_shape = input_buf.shape
            graph.replace(node.op, op_adapter.ReshapeOp(str(node.op.name), shape=input_shape))


@register_layer_optimization
class OptimizeInstanceNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InstanceNormOp.TRANSLATION_KEY
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if 1 < input_buf.rank() <= 5:
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            output_buffer = graph.get_output_buffers(node)[0]
            # (image/feature)_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
            # Enforce the output format here to be NDHWC/NSC/NFC
            output_buffer.axis_format = AxisOrder().get_axis_format(len(output_buffer.shape))
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_INSTANCE_NORM_DIM_UNSUPPORTED")(input_buf.rank(),
                                                                                                      node.op.name))
        return True

    def prepare_inputs_as_params(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        bias_buffer = graph.get_buffer(node.input_names[2])
        bias_node = bias_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
            node.op.bias = bias_node.op.tensor
            # Remove the weights/bias inputs from the IR graph
            graph.remove_node_as_consumer(node, weights_buffer.name)
            graph.remove_node_as_consumer(node, bias_buffer.name)
            node.input_names = [node.input_names[0]]


@register_layer_optimization
class OptimizeL2NormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.L2NormOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_axis_formats = graph.get_input_axis_formats(node)

        if not super(OptimizeL2NormTranslation, self).axes_to_spatial_first_order(node, graph):
            # No change in input formats
            return False

        # transform axis to the correct index, also ensures axis is always positive
        input_buf = graph.get_input_buffers(node)[0]
        if (input_axis_formats[0] == AxisTracker.AxisFormat.NDHWC and \
            node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW) or \
            (input_axis_formats[0] == AxisTracker.AxisFormat.NSC and \
             node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS) or \
            (input_axis_formats[0] == AxisTracker.AxisFormat.NFC and \
             node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF):
            axis_map = graph.src_axis_order.permute_sequence_from_ir[input_buf.rank() - 1]
            if type(node.op.axis) is np.ndarray:
                for i in range(len(node.op.axis)):
                    node.op.axis[i] = axis_map[node.op.axis[i]]
            else:
                node.op.axis = axis_map[node.op.axis]

        return True

    def merge_low_level_ops_to_layers(self, graph):
        def validate(nodes_tuple):
            # Reduce_l2 can be matched to L2Norm only if input to ReduceL2 is also one of the inputs to Div
            reduce_l2_node = nodes_tuple[0]
            div_node = nodes_tuple[-1]
            reduce_l2_input_name = reduce_l2_node.input_names[0]
            if not reduce_l2_input_name in div_node.input_names:
                return False
            # reduce_l2 is reduced to l2_norm only if keep_dims is True
            if not reduce_l2_node.op.keep_dims:
                return False
            return True

        sequence1 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_l2", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_EXPAND, "ALL")])
             ),
            (ir_graph.IR_OP_EXPAND,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [(ir_graph.IR_OP_EXPAND, 1)]), ())
        ]

        sequence2 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_l2", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 1)]), ())
        ]

        sequence3 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [("reduce_l2", 1)]), ())
        ]

        sequence4 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_NEURON, "ALL")])
             ),
            (ir_graph.IR_OP_NEURON, # clip
             ("MATCH_NUM_BUFS", [("reduce_l2", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_EXPAND, "ALL")])
             ),
            (ir_graph.IR_OP_EXPAND,
             ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_NEURON, "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [(ir_graph.IR_OP_EXPAND, 1)]), ())
        ]

        sequences = [sequence1, sequence2, sequence3, sequence4]

        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate, ignore_constants=True)

            for node_tuple in matched_node_list:
                # L2 Norm lower sequence found
                # Check for either sequence 1 or sequence 2 found
                if len(node_tuple) > 3:
                    reduce_l2_node, epsilon_node, _, div_node = node_tuple
                elif len(node_tuple) > 2:
                    reduce_l2_node, epsilon_node, div_node = node_tuple
                else:
                    reduce_l2_node, div_node = node_tuple
                    epsilon_node = None

                l2norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.L2NormOp.type,
                                                                         op_adapter.L2NormOp.LEGACY_TRANSLATION_KEY)
                if epsilon_node:
                    if isinstance(epsilon_node.op, op_adapter.NeuronOp):
                        # get epsilon from clip min
                        elementwise_sum_constant_tensor = epsilon_node.op.min_value
                    else:
                        # get epsilon from elementwise_sum constant op and assign epsilon to new L2normOp
                        elementwise_sum_constant_tensor = graph.get_producer_op(epsilon_node.input_names[1]).tensor
                    l2norm_op = op_adapter.L2NormOp(l2norm_op_name, axis=-1, epsilon=elementwise_sum_constant_tensor)
                else:
                    # epsilon_node is not present
                    l2norm_op = op_adapter.L2NormOp(l2norm_op_name, axis=-1)

                # Prune all matched nodes except Last node
                for node in node_tuple[:-1]:
                    graph.prune(node, force_remove=True)

                # Replace Last op with L2norm. No need to connect input of first node in pattern to the last_node
                # since Div node already has that as the other input
                last_op = node_tuple[-1].op
                graph.replace(last_op, l2norm_op)


@register_layer_optimization
class OptimizeLayerNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LayerNormOp.TRANSLATION_KEY
        self.register_method(MATCH_LAYERNORM, self.match_layer_norm)

    @staticmethod
    def match_layer_norm(graph):
        sequence1 = [
            ("reduce_mean",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_power", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_power",
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_power", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequence2 = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("Reshape", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_power", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_power",
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("elementwise_power", "ALL")]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("Reshape", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequences = [sequence1, sequence2]
        for idx, sequence in enumerate(sequences):
            matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)

            for node_tuple in matched_node_list:
                if idx == 0:
                    reduce_mean_node = node_tuple[0]
                    prunable_nodes = node_tuple[::-1]
                else:
                    reshape_node = node_tuple[0]
                    reshape_node_input_bufs = graph.get_input_buffers(reshape_node)
                    reduce_mean_node = node_tuple[1]
                    # return nodes in reverse order except the first reshape node
                    prunable_nodes = node_tuple[:0:-1]

                beta_input_name = [name for name in node_tuple[-1].input_names if
                                   (isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                gamma_input_name = [name for name in node_tuple[-2].input_names if
                                    (isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                axes = [0]
                epsilon = op_adapter.LayerNormOp.EPSILON

                last_node = node_tuple[-1]
                last_node_buf = graph.get_output_buffers(last_node)
                last_node_consumers = last_node_buf[0].consumers
                last_node_consumers_names = [node.op.name for node in last_node_consumers]
                # maps consumers of last node buffer with corresponding input_names
                last_node_consumers_input_names = {}
                for consumer in last_node_consumers:
                    last_node_consumers_input_names[consumer] = copy.deepcopy(consumer.input_names)

                # Prune all matched nodes in reverse order
                for node in prunable_nodes:
                    input_names = node.input_names[:]
                    # determine axes parameter of LayerNorm from ReduceMean Op
                    if node.op.type == op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN]:
                        axes = node.op.axes
                    # determine epsilon parameter of LayerNorm from the ElementwiseSumOp Op with constant input of
                    # size 1
                    if node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]:
                        input_name = [name for name in input_names if
                                      (isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                        if graph.get_producer_op(input_name).tensor.size == 1:
                            epsilon = graph.get_producer_op(input_name).tensor[0]

                    graph.prune(node, force_remove=True)

                # reassign input_names to consumers of last node buffer post pruning of the nodes
                for consumer in last_node_consumers:
                    consumer.input_names = last_node_consumers_input_names[consumer]

                layer_norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.LayerNormOp.type,
                                                                             op_adapter.LayerNormOp.LEGACY_TRANSLATION_KEY)
                layer_norm_input_names = reduce_mean_node.input_names + [gamma_input_name, beta_input_name]
                if idx == 0:
                    layer_norm_output_names = last_node.output_names
                else:
                    layer_norm_output_names = [layer_norm_op_name]

                layer_norm_op = op_adapter.LayerNormOp(layer_norm_op_name, axes=axes, epsilon=epsilon)

                # compute the correct idx to insert layer_norm
                idx_to_insert = 0
                for input_name in layer_norm_input_names:
                    buf = graph.get_buffer(input_name)
                    cur_idx = graph.nodes_in_order.index(buf.producer)
                    if idx_to_insert < cur_idx:
                        idx_to_insert = cur_idx

                layer_norm_node = graph.add(layer_norm_op, input_names=layer_norm_input_names, output_names=layer_norm_output_names,
                                            idx=idx_to_insert+1)

                # add consumers of layer_norm output buffer
                for output_name in layer_norm_output_names:
                    output_buf_ = graph.get_buffer(output_name)
                    output_buf_.consumers = last_node_consumers

                if idx == 1:
                    # add reshape node after layer_norm
                    reshape_name = layer_norm_op_name + "_postprocess_reshape"
                    reshape_op = op_adapter.ReshapeOp(reshape_name, shape=reshape_node_input_bufs[0].shape)
                    reshape_node = graph.inject(reshape_op, input_name=layer_norm_node.output_names[0], output_name=last_node.output_names[0],
                                                consumer_names=last_node_consumers_names if last_node_consumers_names else None)
                    graph.replace_quantization_param(last_node.op.name, reshape_name)
                else:
                    graph.replace_quantization_param(last_node.op.name, layer_norm_node.op.name)

    def axes_to_spatial_first_order(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):

        input_buffers = graph.get_input_buffers(node)
        if node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCDHW,
                                            AxisTracker.AxisFormat.NONTRIVIAL] and \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NDHWC:
            permute_order = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
            graph.inject_implicit_permute(input_buffers[0].name,
                                          AxisTracker.AxisFormat.NCDHW,
                                          permute_order,
                                          [node.op.name])
        elif node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NONTRIVIAL] and \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NSC:
            permute_order = AxisTracker.AxisFormat.NSC_TO_NCS
            graph.inject_implicit_permute(input_buffers[0].name,
                                          AxisTracker.AxisFormat.NCS,
                                          permute_order,
                                          [node.op.name])
        elif node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NONTRIVIAL] and \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NFC:
            permute_order = AxisTracker.AxisFormat.NFC_TO_NCF
            graph.inject_implicit_permute(input_buffers[0].name,
                                          AxisTracker.AxisFormat.NCF,
                                          permute_order,
                                          [node.op.name])
        elif input_buffers[0].axis_format == AxisTracker.AxisFormat.NONTRIVIAL and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NONTRIVIAL:
            # No change needed
            return False
        elif input_buffers[0].axis_format == AxisTracker.AxisFormat.NCDHW or \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NCS or \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NCF or \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NF or \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.NC or \
                input_buffers[0].axis_format == AxisTracker.AxisFormat.ANY:
            # No change needed
            return False
        else:
            raise ValueError("LayerNorm Op {}: Got Unknown input axis format {} data axis format {} input shape {}".
                format(
                    node,
                    input_buffers[0].axis_format,
                    node.op.data_axis_formats[0],
                    input_buffers[0].shape
                  ))

        return True


@register_layer_optimization
class OptimizeLogSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LogSoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeLogSoftmaxTranslation, self).axes_to_spatial_first_order(node, graph):
            return False

        # Ensure we're using the correct input buffer as a permute might have been inserted above
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            axis_map = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            log_debug('Mapping axis from {} to {}: '.format(node.op.axis, axis_map[node.op.axis]))
            node.op.axis = axis_map[node.op.axis]
        return True


@register_layer_optimization
class OptimizeLrnTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LrnOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMatmulTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MatMulOp.TRANSLATION_KEY
        self.register_method(ALIGN_MATMUL_RANKS, self.align_matmul_input_ranks)

    @staticmethod
    def align_matmul_input_ranks(node, graph):
        inputs = graph.get_input_buffers(node)
        output = graph.get_output_buffers(node)[0]
        log_debug1("Running matmal optimization for {}".format(node.op.name))
        if inputs[0].rank() != inputs[1].rank():
            log_debug1("Matmul {} input {} rank {} != input2 {} rank {}".format(node.op.name, inputs[0].name, inputs[0].rank(), inputs[1].name, inputs[1].rank()))
            lower_rank_input_buf, larger_rank_input_buf = (inputs[0], inputs[1]) \
                            if inputs[0].rank() < inputs[1].rank() else (inputs[1], inputs[0])

            # Adding reshape nodes to expand rank to match other input
            producer = lower_rank_input_buf.producer.op
            new_shape = translation_utils.expand_to_rank(lower_rank_input_buf.shape, len(larger_rank_input_buf.shape))
            log_debug1("This matmul impl requires identical rank, reshaping {} to {}".format(lower_rank_input_buf.shape, new_shape))
            if producer.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                producer.tensor = producer.tensor.reshape(new_shape)
                lower_rank_input_buf.shape = new_shape
                lower_rank_input_buf.axis_format = larger_rank_input_buf.axis_format
            else:
                reshape_node_name = output.name + "_" + lower_rank_input_buf.name + "_reshape"
                reshape_op = op_adapter.ReshapeOp(name=reshape_node_name,
                                                  shape=new_shape)
                graph.inject(reshape_op, input_name=lower_rank_input_buf.name,
                             output_name=reshape_node_name, consumer_names=[node.op.name],
                             axis_format=larger_rank_input_buf.axis_format)
        # Reevaluate input buffers since reshape may have been added
        inputs = graph.get_input_buffers(node)
        node.op.populate_data_axis_formats(graph, inputs)

    def axes_to_spatial_first_order(self, node, graph):
        # matmul is always performed in Src Framework order,
        # because only then the last 2 dimensions will align
        for i, input_name in enumerate(node.input_names):
            input_buf = graph.get_buffer(input_name)
            if input_buf.axis_format == node.op.data_axis_formats[i] and \
                    input_buf.axis_format in spatial_first_format_to_channel_first_format:
                input_buf.axis_format = spatial_first_format_to_channel_first_format[input_buf.axis_format]
            elif input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NCDHW:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NCS:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NCF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NC or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_MATMUL_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = graph.get_input_buffers(node)[0].axis_format

        return True

    def merge_low_level_ops_to_layers(self, graph):
        def validate_dims(nodes_tuple):
            matmul_node = nodes_tuple[0]
            elementwise_sum_node = nodes_tuple[1]
            elementwise_sum_input_nodes = graph.get_op_input_nodes(elementwise_sum_node)
            constant_bias_node = [node for node in elementwise_sum_input_nodes if node.op.TRANSLATION_KEY == 'constant'][0]
            matmul_input_buffers = graph.get_input_buffers(matmul_node)
            matmul_input_0_buffer = matmul_input_buffers[0]
            matmul_input_1_buffer = matmul_input_buffers[1]
            bias_buffer = graph.get_output_buffers(constant_bias_node)[0]

            if matmul_node.output_names[0] in elementwise_sum_node.input_names \
                    and matmul_input_0_buffer.shape[-1] == matmul_input_1_buffer.shape[-2] \
                    and np.prod(bias_buffer.shape) == bias_buffer.shape[-1] \
                    and matmul_input_1_buffer.shape[-1] == bias_buffer.shape[-1]:
                return True
            return False

        def validate_sequence2(nodes_tuple):
            matmul_node = nodes_tuple[0]
            matmul_input_buffers = graph.get_input_buffers(matmul_node)
            matmul_input_0_buffer = matmul_input_buffers[0]
            matmul_input_1_buffer = matmul_input_buffers[1]

            if matmul_input_0_buffer.shape[-1] == matmul_input_1_buffer.shape[-2] \
                    and matmul_input_1_buffer.rank() == 2:
                return True
            return False

        def change_buffer_name(old_buffer_name, new_buffer_name):
            # Update the name of the buffer
            node_output_buffer = graph.get_buffer(old_buffer_name)
            prev_node = node_output_buffer.producer
            node_output_buffer.name = new_buffer_name
            graph.buffers[new_buffer_name] = node_output_buffer
            # Update the consumer nodes.
            for consumer in node_output_buffer.consumers:
                in_idx = consumer.input_names.index(old_buffer_name)
                consumer.input_names[in_idx] = new_buffer_name
            # Update the output names.
            out_idx = prev_node.output_names.index(old_buffer_name)
            prev_node.output_names[out_idx] = new_buffer_name
            del graph.buffers[old_buffer_name]

        sequence1 = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]), ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("constant", "ANY"), ("MatMul", "ANY")]), ()
             )
        ]

        sequence2 = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
             ()
             )
        ]
        sequences = [sequence1, sequence2]
        for idx, sequence in enumerate(sequences):
            if idx == 0:
                matched_node_list = graph.get_matched_nodes(sequence, validator=validate_dims, ignore_constants=True)
            else:
                matched_node_list = graph.get_matched_nodes(sequence, validator=validate_sequence2)
            for node_tuple in matched_node_list:
                matmul_node = node_tuple[0]
                constant_weights_node = graph.get_op_input_nodes(matmul_node)[1]
                if idx == 0:
                    elementwise_sum_node = node_tuple[1]
                    elementwise_input_nodes = graph.get_op_input_nodes(elementwise_sum_node)
                    constant_bias_node = [node for node in elementwise_input_nodes if node.op.TRANSLATION_KEY == 'constant'][0]

                matmul_input_buffer = graph.get_input_buffers(matmul_node)[0]
                pre_reshape_name = matmul_node.op.name + "_pre_reshape"
                pre_reshape_shape = [np.prod(matmul_input_buffer.shape[:-1]), matmul_input_buffer.shape[-1]]
                pre_reshape_op = op_adapter.ReshapeOp(pre_reshape_name, shape=pre_reshape_shape)
                graph.inject(pre_reshape_op, input_name=matmul_node.input_names[0], output_name=pre_reshape_name,
                            consumer_names=[matmul_node.op.name], axis_format=AxisTracker.AxisFormat.NF)
                matmul_weights = constant_weights_node.op.tensor
                matmul_op = matmul_node.op
                matmul_op_name = matmul_op.name
                fc_op_name = matmul_op_name + '_fc'

                if idx == 0:
                    bias = constant_bias_node.op.tensor.copy()
                    constant_bias_node.op.tensor = np.atleast_1d(np.squeeze(bias))
                    bias_name = constant_bias_node.op.name
                else:
                    bias_tensor = np.zeros(matmul_weights.shape[-1], dtype=np.float32)
                    bias_op = op_adapter.ConstantOp(constant_weights_node.op.name + '_b', tensor=bias_tensor.copy())
                    bias_name = bias_op.name
                    matmul_idx = graph.list_nodes().index(matmul_node)
                    graph.add(bias_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY], idx=matmul_idx)

                fc_op = op_adapter.FullyConnectedOp(name=fc_op_name, bias_op_name=bias_name)
                graph.replace(matmul_op, fc_op)
                fc_node = graph.get_node_by_name(fc_op.name)
                bias_buffer = graph.get_buffer(bias_name)
                bias_buffer.consumers.add(fc_node)
                fc_node.input_names.append(bias_name)
                # update data_axis_formats of the fc_node
                fc_node.op.data_axis_formats.append(bias_buffer.axis_format)

                if idx == 0:
                    bias_buffer.consumers.remove(elementwise_sum_node)
                    elementwise_sum_node.input_names.remove(bias_name)

                    # update order from [input, weight, fully_connected, bias] to [input, weight, bias, fully_connected]
                    idx_bias = graph.nodes_in_order.index(constant_bias_node)
                    idx_fc = graph.nodes_in_order.index(fc_node)
                    if idx_bias > idx_fc:
                        graph.nodes_in_order[idx_fc] = constant_bias_node
                        graph.nodes_in_order[idx_bias] = fc_node

                    graph.squash(elementwise_sum_node, graph.get_output_buffers(fc_node)[0].name)

                    elementwise_sum_output_buffer = graph.get_output_buffers(elementwise_sum_node)[0]
                    # Refetch fc_node after squash
                    fc_node = graph.get_node_by_name(fc_op.name)
                    # Transferring activation encoding of elementwise sum to fullyconnected.
                    q = graph.user_quantization_overrides
                    if q and 'activation_encodings' in q and elementwise_sum_output_buffer.name in q['activation_encodings']:
                        activations = q['activation_encodings']
                        act_encs = [IROptimizations.extract_encoding_dict(graph.get_output_buffers(fc_node)[0].name,
                                                                        activations[elementwise_sum_output_buffer.name])]
                        graph.add_quantization_params(fc_node.op.name, output_encodings=act_encs)

                    # adds the quantization params to the graph if present
                    q = graph.user_quantization_overrides
                    if q and 'param_encodings' in q and bias_name in q['param_encodings']:
                        params = q['param_encodings']
                        param_encs = [IROptimizations.extract_encoding_dict('bias', params[bias_name])]
                        graph.add_quantization_params(fc_op_name, param_encodings=param_encs)

                    fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                    old_buffer_name = fc_node_output_buffer.name
                    new_buffer_name = fc_node_output_buffer.name + '_fc'
                    change_buffer_name(old_buffer_name, new_buffer_name)

                    post_reshape_name = fc_node.op.name + '_post_reshape'
                    post_reshape_op = op_adapter.ReshapeOp(post_reshape_name,
                                                        shape=elementwise_sum_output_buffer.shape[:])
                    graph.inject(post_reshape_op, new_buffer_name, old_buffer_name,
                                axis_format=matmul_input_buffer.axis_format)
                else:
                    fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                    old_buffer_name = fc_node_output_buffer.name
                    new_buffer_name = fc_node_output_buffer.name + '_fc'
                    change_buffer_name(old_buffer_name, new_buffer_name)
                    fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                    post_reshape_name = fc_node.op.name + '_post_reshape'
                    post_reshape_shape = [*(matmul_input_buffer.shape[:-1]), matmul_weights.shape[-1]]
                    post_reshape_op = op_adapter.ReshapeOp(post_reshape_name,
                                                        shape=post_reshape_shape)
                    graph.inject(post_reshape_op, new_buffer_name, old_buffer_name,
                                axis_format=matmul_input_buffer.axis_format)

                # Refetch the fc_node to update the output buffer shape to 2D
                fc_node = graph.get_node_by_name(fc_op.name)
                fc_out_buf = graph.get_output_buffers(fc_node)[0]
                # np.prod assigns default platform integer int64
                # which is not consistent to the default int. So the dtypes we get are int64 and int
                # Cast to int() makes it consistent (int, int)
                fc_out_buf.shape = [int(np.prod(matmul_input_buffer.shape[:-1])), matmul_weights.shape[-1]]


@register_layer_optimization
class OptimizeNeuronTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NeuronOp.TRANSLATION_KEY
        self.register_method(MATCH_HARDSWISH, self.match_hardswish)

    @staticmethod
    def match_hardswish(graph):
        def is_valid_hardswish(node_tuple):
            def check_for_valid_add_node(input_const_name):
                const_input_node = graph.get_producer_node(input_const_name)
                const_input_value = const_input_node.op.tensor
                const_input_length = reduce(lambda x,y:x * y, const_input_value.shape)
                temp = set(const_input_value.reshape(const_input_length))
                if len(temp) != 1 or int(temp.pop()) != 3:
                    return False
                return True

            def check_for_valid_neuron_node(node):
                if node.op.neuron_type != ir_graph.QNN_OP_RELU_MIN_MAX \
                        or int(node.op.min_value) != 0 \
                        or int(node.op.max_value) != 6:
                    return False
                return True

            def check_for_valid_div_node(node):
                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if np.array_equal(np.unique(const_input_value), [6]):
                  return True
                return False

            def check_for_valid_mul_node_with_const_input(node):
                def is_close_to_one_sixth(num):
                    return translation_utils.compare_values(float(num[0]), 1/6, rtol=1.e-3, atol=1.e-5)

                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if const_input_value.shape != (1,) or not is_close_to_one_sixth(const_input_value):
                    return False
                return True

            add_node, neuron_node = node_tuple[0], node_tuple[1]
            add_non_const_input_name, add_const_input_name, mul_node, mul_node_const_input, div_node = [None] * 5
            for input_name in add_node.input_names:
                if graph.get_producer_op(input_name).type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    add_const_input_name = input_name
                else:
                    add_non_const_input_name = input_name

            for node in node_tuple[2:]:
                if node.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE]:
                    div_node = node
                else:
                    mul_input_names = node.input_names
                    if len(mul_input_names) != 2:
                        return False
                    if any(op_adapter.ConstantOp.TRANSLATION_KEY == graph.get_producer_op(input_name).type
                           for input_name in mul_input_names):
                        mul_node_const_input = node
                    else:
                        mul_node = node

            if not add_const_input_name or not mul_node or (not div_node and not mul_node_const_input):
                return False

            if add_non_const_input_name not in mul_node.input_names:
                # the add and mul must share same input_name to be matched as hswish
                return False

            return (check_for_valid_add_node(add_const_input_name) and
                    check_for_valid_neuron_node(neuron_node) and
                    (check_for_valid_div_node(div_node) if div_node else
                     check_for_valid_mul_node_with_const_input(mul_node_const_input)))

        def get_input_const_nodes(input_names):
            input_nodes = [graph.buffers[name].producer for name in input_names]
            const_nodes = [node for node in input_nodes if
                           node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY]
            return const_nodes

        def remove_const_nodes(node_tuple, matched_sequence_flag):
            if matched_sequence_flag[-1] in ['1', '3']:
                nodes_with_const_input = [node_tuple[0], node_tuple[3]]
            else:
                nodes_with_const_input = [node_tuple[0], node_tuple[2]]

            for node in nodes_with_const_input:
                const_node = get_input_const_nodes(node.input_names)[0]
                const_node_output_buf = graph.get_buffer(const_node.output_names[0])
                if len(const_node_output_buf.consumers) == 1:
                    # Only prune const_node if node is its only consumer
                    graph.prune(const_node, force_remove=True)
                else:
                    # Else, disconnect from node and leave const_node alone
                    const_node_output_buf.consumers.remove(node)
                    node.input_names.remove(const_node_output_buf.name)

        # Y = X*RELU6(X+3)*(1/6) or X*CLIP(X+3)*(1/6)
        sequence1 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("Neuron", "ALL")])
             ),
            ("Neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)*(1/6)) or X*(CLIP(X+3)*(1/6))
        sequence2 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("Neuron", "ALL")])
             ),
            ("Neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("Neuron", "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ()
             )
        ]

        # Y = X*RELU6(X+3)/6 or X*CLIP(X+3)/6
        sequence3 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("Neuron", "ALL")])
             ),
            ("Neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)/6) or X*(CLIP(X+3)/6)
        sequence4 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [("Neuron", "ALL")])
             ),
            ("Neuron",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("Neuron", "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ()
             )
        ]

        sequences = [sequence1, sequence2, sequence3, sequence4]

        for index, sequence in enumerate(sequences):
            matched_sequence_flag = 'matched_sequence' + str(index + 1)
            matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_hardswish, ignore_constants=True)

            for node_tuple in matched_node_list:
                remove_const_nodes(node_tuple, matched_sequence_flag)
                add_node = node_tuple[0]
                for node in node_tuple[:0:-1]:
                    for input_name in node.input_names:
                        if len(graph.get_buffer(input_name).consumers) == 1:
                            # per the sequence matching we know one of the inputs are squashable, hence
                            # check which either one has 1 consumer
                            graph.squash(node, input_name=input_name)

                add_op = add_node.op
                add_op_name = add_op.name
                hardswish_op_name = add_op_name + '_Hswish'
                hardswish_op = op_adapter.NeuronOp(hardswish_op_name, ir_graph.QNN_OP_HARD_SWISH)
                graph.replace(add_op, hardswish_op)

    def check_static_equal_tensor_vals_input(self, node, graph):
        perform_optimization = True
        if graph.get_producer_op(node.input_names[0]).type == op_adapter.ConstantOp.TRANSLATION_KEY:
            tensor = graph.get_producer_op(node.input_names[0]).tensor
            constant_node = graph.get_node_by_name(node.input_names[0])
        elif graph.get_producer_op(node.input_names[1]).type == op_adapter.ConstantOp.TRANSLATION_KEY:
            tensor = graph.get_producer_op(node.input_names[1]).tensor
            constant_node = graph.get_node_by_name(node.input_names[1])
        else:
            perform_optimization = False

        if perform_optimization:
            # Perform some more verification on min and max clips by flattening and checking if all vals are the same
            # Get flattened 1D array for multidimensional tensor array
            flatten_arr = np.ravel(tensor)
            # Check if all value in min_val array are equal
            perform_optimization = np.all(tensor == flatten_arr[0])
            tensor_scalar_value = None
            if perform_optimization:
                tensor_scalar_value = flatten_arr[0]

        return perform_optimization, constant_node, tensor_scalar_value

    def relu_min_max_sequence_optimization(self, node_replace, node_delete, relu_min_max_op, graph):
        # Replace elementwise min or elementwise max op depending on sequence found with reluminmax op
        graph.replace(node_replace.op, relu_min_max_op)

        # Get output buffer of last op in sequence
        node_delete_output_buff_name = node_delete.output_names
        node_delete_buff = graph.get_buffer(node_delete_output_buff_name[0])

        # Update input_names of consumers of elementwise min or elementwise max output buffer depending on sequence
        node_delete_buff_consumers = node_delete_buff.consumers
        for consumer in node_delete_buff_consumers:
            # consumer.input_names = graph.nodes_by_name[relu_min_max_op.name].output_names
            for idx in range(len(consumer.input_names)):
                if consumer.input_names[idx] == node_delete_output_buff_name[0]:
                    consumer.input_names[idx] = graph.nodes_by_name[relu_min_max_op.name].output_names[0]
        node_delete_buff.consumers = set()
        graph.prune(node_delete, force_remove=True)
        graph.get_buffer(node_replace.output_names[0]).consumers = node_delete_buff_consumers

    def merge_low_level_ops_to_layers(self, graph):

        def validate_node(node_tuple):
            if node_tuple[0].op.eltwise_type == ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM:
                min_node, max_node = node_tuple
            else:
                max_node, min_node = node_tuple

            # Check to see if elementwise min and max have one constant input
            # Verify if tensor values in min and max clip arrays are all same for optimization to work properly
            perform_optimization, check_tensor_vals_equal, _ = self.check_static_equal_tensor_vals_input(min_node, graph)
            perform_optimization_2, check_tensor_vals_equal_2, _ = self.check_static_equal_tensor_vals_input(max_node, graph)

            if not perform_optimization or not perform_optimization_2:
                return False
            return True

        # Elementwisemin -> Elementwisemax = Reluminmax
        sequence_1 = [
            ("elementwise_min",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_max", "ALL")])
             ),
            ("elementwise_max",
             ("MATCH_NUM_BUFS", [("elementwise_min", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence_2 = [
            ("elementwise_max",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_min", "ALL")])
             ),
            ("elementwise_min",
             ("MATCH_NUM_BUFS", [("elementwise_max", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequences = [sequence_1, sequence_2]
        for idx, sequence in enumerate(sequences):
            matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True, validator=validate_node)
            for node_tuple in matched_node_list:
                is_min_max_sequence = False
                if node_tuple[0].op.eltwise_type == ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM:
                    is_min_max_sequence = True

                if is_min_max_sequence:
                    min_node, max_node = node_tuple
                else:
                    max_node, min_node = node_tuple

                # Retrieve tensor and node information required for ReluMinMax optimization
                _, constant_node_min, min_op_scalar_value = self.check_static_equal_tensor_vals_input(min_node, graph)
                _, constant_node_max, max_op_scalar_value = self.check_static_equal_tensor_vals_input(max_node, graph)

                if max_op_scalar_value <= min_op_scalar_value:
                    # if min is greater than or equal to max, then assign min value(eltwise min) as max for ReluMinMax and
                    # max value(eltwise max)  as min for ReluMinMax
                    relu_max_value, relu_min_value = min_op_scalar_value, max_op_scalar_value
                else:
                    if is_min_max_sequence:
                        # When elementwise min is followed by elementwise max and min value is less than max value,
                        # assign both min and max for ReluMinMax to max value(elementwise max)
                        relu_min_value = max_op_scalar_value
                        relu_max_value = max_op_scalar_value
                    else:
                        # When elementwise max is followed by elementwise min and min value is less than max value,
                        # assign both min and max for ReluMinMax to min value(elementwise min)
                        relu_min_value = min_op_scalar_value
                        relu_max_value = min_op_scalar_value

                # Assign values of old sequence ops to new op
                relu_min_max_op = op_adapter.NeuronOp("", ir_graph.QNN_OP_RELU_MIN_MAX)
                relu_min_max_op.name = graph.naming_policy.get_op_name(relu_min_max_op)
                relu_min_max_op.min_value = relu_min_value
                relu_min_max_op.max_value = relu_max_value

                # Check to remove constant node from graph if consumer of it is only first node in sequence matched.
                # Second node's constant input will be automatically removed by remove_disconnected_nodes optimization
                # since second node in sequence is pruned
                if is_min_max_sequence:
                    # removing node as consumer should take care of removing the min node as a consumer of the constant
                    # regardless of however many consumers the constant node has besides the min node
                    graph.remove_node_as_consumer(min_node, graph.get_buffer(constant_node_min.output_names[0]).name)
                    self.relu_min_max_sequence_optimization(min_node, max_node, relu_min_max_op, graph)
                else:
                    # removing node as consumer should take care of removing the max node as a consumer of the constant
                    # regardless of however many consumers the constant node has besides the max node
                    graph.remove_node_as_consumer(max_node, graph.get_buffer(constant_node_max.output_names[0]).name)
                    self.relu_min_max_sequence_optimization(max_node, min_node, relu_min_max_op, graph)


@register_layer_optimization
class OptimizeNonZeroTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NonZeroOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        # input buffer should be in source framework order.
        input_buf = graph.get_input_buffers(node)[0]
        data_axis_format = node.op.data_axis_formats[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                data_axis_format == AxisTracker.AxisFormat.NCDHW:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NCDHW,
                                            AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                data_axis_format == AxisTracker.AxisFormat.NCS:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NCS,
                                            AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                data_axis_format == AxisTracker.AxisFormat.NCF:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NCF,
                                            AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                data_axis_format == AxisTracker.AxisFormat.TNF:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.TNF,
                                          AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        return True


@register_layer_optimization
class OptimizeOneHotTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.OneHotOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizePadTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PadOp.TRANSLATION_KEY
        self.register_method(SQUASH_PAD, self.squash_pad)

    @staticmethod
    def squash_pad(graph):
        def validate_node(nodes_tuple):
            pad_node_ = nodes_tuple[0]
            pads = pad_node_.op.pad_amount
            # squash if all values are 0s
            if all(not (pad_0 or pad_1) for pad_0, pad_1 in pads) and \
                    len(graph.get_buffer(pad_node_.input_names[0]).consumers) == 1:
                return True
            return False

        sequence = [
            ("Pad", (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            pad_node = node_tuple[0]
            graph.squash_identity(pad_node)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.NCF_TO_NFC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.TNF_TO_NTF)
        node.op.pad_amount = np.asarray(node.op.pad_amount, dtype=np.dtype('uint32'))
        return True


@register_layer_optimization
class OptimizePoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Pool2dOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_buffers = graph.get_input_buffers(node)
        input_axis_formats = [buf.axis_format for buf in input_buffers]

        if any(axis_format in input_axis_formats for axis_format in [AxisTracker.AxisFormat.NDHWC,
                                                                     AxisTracker.AxisFormat.NCDHW,
                                                                     AxisTracker.AxisFormat.NSC,
                                                                     AxisTracker.AxisFormat.NCS,
                                                                     AxisTracker.AxisFormat.ANY,
                                                                     AxisTracker.AxisFormat.NONTRIVIAL]):
            AxisTracker.image_to_channel_last_order(node, graph)
            output_buffer = graph.get_output_buffers(node)[0]
            # image_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
            # Enforce the output format here according to output buffer's rank
            output_buffer.axis_format = AxisOrder().get_axis_format(output_buffer.rank())
        else:
            raise ValueError("Pool Node {} got unexpected input_axis_formats {}".format(node, input_axis_formats))
        return True


@register_layer_optimization
class OptimizePool1DTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.Pool1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        pool_op_name = node.op.name + "_2d"
        if node.op.pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
            self.nn_2d_op = op_adapter.Pool2dOp(pool_op_name,
                                                pool_type=node.op.pool_type,
                                                size_y=node.op.filter_size,
                                                size_x=1,
                                                stride_y=node.op.stride,
                                                stride_x=1,
                                                pady_before=node.op.pad_amount[0],
                                                pady_after=node.op.pad_amount[1],
                                                padx_before=0,
                                                padx_after=0,
                                                padding_size_strategy=node.op.padding_size_strategy)
        elif node.op.pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
            self.nn_2d_op = op_adapter.Pool2dOp(pool_op_name,
                                                pool_type=node.op.pool_type,
                                                size_y=node.op.filter_size,
                                                size_x=1,
                                                stride_y=node.op.stride,
                                                stride_x=1,
                                                pady_before=node.op.pad_amount[0],
                                                pady_after=node.op.pad_amount[1],
                                                padx_before=0,
                                                padx_after=0,
                                                padding_size_strategy=node.op.padding_size_strategy,
                                                count_pad_for_edges=node.op.count_pad_for_edges)
        elif node.op.pool_type == ir_graph.QNN_OP_L2_POOL_2D:
            self.nn_2d_op = op_adapter.Pool2dOp(pool_op_name,
                                                pool_type=node.op.pool_type,
                                                size_y=node.op.filter_size,
                                                size_x=1,
                                                stride_y=node.op.stride,
                                                stride_x=1,
                                                pady_before=node.op.pad_amount[0],
                                                pady_after=node.op.pad_amount[1],
                                                padx_before=0,
                                                padx_after=0,
                                                padding_size_strategy=node.op.padding_size_strategy,
                                                p=node.op.p)

        super().expand_1d_spatial_nn_nodes(node, graph)


@register_layer_optimization
class OptimizePool3dTranslation(OptimizePoolTranslation):
    def __init__(self):
        OptimizePoolTranslation.__init__(self)
        self.op_type = op_adapter.Pool3dOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeTransposeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TransposeOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        # check for trivial cases first, which will end up
        # in removal. Otherwise, just set output order to nontrivial
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC:
                # This permute changes from NDHWC to NCDHW which is opposite of desired, so skip this node
                if output_buf.axis_format == AxisTracker.AxisFormat.NCDHW and node.op.perm == [0, 4, 1, 2, 3]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                        node.op.perm == [0, 1, 2, 3, 4]:
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
                # This permute changes from NCDHW to NDHWC but input has already changed to NDHWC, so skip
                if output_buf.axis_format == AxisTracker.AxisFormat.NDHWC and node.op.perm == [0, 2, 3, 4, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NCDHW and \
                        node.op.perm == [0, 1, 2, 3, 4]:
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                                  consumers=[node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                # going to nontrivial, hoping for the best.
                log_warning("Op {} with Permute order {}: Unknown input data format {}".format(node,
                                                                                               node.op.perm,
                                                                                               node.op.data_axis_formats[0]))
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCDHW and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NONTRIVIAL]:
            if output_buf.axis_format == AxisTracker.AxisFormat.NDHWC and node.op.perm == [0, 2, 3, 4, 1]:
                return
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NSC:
                # This permute changes from NSC to NCS which is opposite of desired, so skip this node
                if output_buf.axis_format == AxisTracker.AxisFormat.NCS and node.op.perm == [0, 3, 1, 2]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                        node.op.perm == [0, 1, 2, 3]:
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
                # This permute changes from NCS to NSC but input has already changed to NSC, so skip
                if output_buf.axis_format == AxisTracker.AxisFormat.NSC and node.op.perm == [0, 2, 3, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NCS and \
                        node.op.perm == [0, 1, 2, 3]:
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    output_shape = AxisTracker.permute_shape(output_buf.get_buf_dims(), AxisTracker.AxisFormat.NCS_TO_NSC)
                    output_buf.set_buf_dims(output_shape)
                    return
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS,
                                                  consumers=[node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                # going to nontrivial, hoping for the best.
                log_warning("Op {} with Permute order {}: Unknown input data format {}".format(node,
                                                                                               node.op.perm,
                                                                                               node.op.data_axis_formats[0]))
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCS and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NONTRIVIAL]:
            if output_buf.axis_format == AxisTracker.AxisFormat.NSC and node.op.perm == [0, 2, 3, 1]:
                return
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NFC:
                # This permute changes from NFC to NCF which is opposite of desired, so skip this node
                if output_buf.axis_format == AxisTracker.AxisFormat.NCF and node.op.perm == [0, 2, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NFC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                        node.op.perm == [0, 1, 2]:
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            elif node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
                # This permute changes from NCF to NFC but input has already changed to NFC, so skip
                print("--debug--2 node.op.perm:{}, type:{}".format(node.op.perm, type(node.op.perm)))
                if output_buf.axis_format == AxisTracker.AxisFormat.NFC and node.op.perm == [0, 2, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NCF
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NCF and \
                        node.op.perm == [0, 1, 2]:
                    output_buf.axis_format = AxisTracker.AxisFormat.NFC
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                                  AxisTracker.AxisFormat.NFC_TO_NCF,
                                                  consumers=[node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                # going to nontrivial, hoping for the best.
                log_warning("Op {} with Permute order {}: Unknown input data format {}".format(node,
                                                                                               node.op.perm,
                                                                                               node.op.data_axis_formats[0]))
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCF and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCF, AxisTracker.AxisFormat.NONTRIVIAL]:
            if output_buf.axis_format == AxisTracker.AxisFormat.NFC and node.op.perm == [0, 2, 1]:
                return
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF and node.op.perm == [1, 0, 2]:
                node.op.perm = [0, 1, 2]
                output_buf.axis_format = AxisTracker.AxisFormat.NTF
            else:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.TNF_TO_NTF,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker. AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                input_buf.axis_format == AxisTracker.AxisFormat.NF:
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError("Permute Op {} got unexpected params: input format {} saved data format {}".format(
                             node, input_buf.axis_format, node.op.data_axis_formats[0]))

        return True

    @staticmethod
    def remove_identity(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and node.op.perm == list(range(len(node.op.perm))):
            # this permute is trivial, remove it
            graph.squash(node, input_name=input_buffer.name)
        return True


@register_layer_optimization
class OptimizePreluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PreluOp.TRANSLATION_KEY
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    @classmethod
    def _permute_coeff(cls, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        coeff_buf = graph.get_buffer(node.input_names[1])
        coeff_shape = coeff_buf.shape

        # determine the permute order(if any) after spatial first transformation
        # Note: only NDHWC, NSC, NFC, and NTF formats imply permute was done.
        input_permute_order = None
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            input_permute_order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            input_permute_order = AxisTracker.AxisFormat.NCS_TO_NSC
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            input_permute_order = AxisTracker.AxisFormat.NCF_TO_NFC
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            input_permute_order = AxisTracker.AxisFormat.TNF_TO_NTF

        if len(coeff_buf.shape) != 1 and len(coeff_buf.shape) != len(input_buf.shape):
            raise ValueError("Prelu coefficient rank must equal either 1 or input rank {} for node {}. Got {} instead."
                             .format(len(input_buf.shape), node.op.name, len(coeff_buf.shape)))
        if input_permute_order is not None and len(coeff_shape) > 1:
            # The input has been permuted hence we also need to permute coeff so that broadcasting persists
            coeff_buf.producer.op.tensor = np.ascontiguousarray(np.transpose(coeff_buf.producer.op.tensor, input_permute_order))
            coeff_shape = coeff_buf.producer.op.tensor.shape

        if not translation_utils.broadcastable(input_buf.shape, coeff_shape):
            raise ValueError(code_to_message.get_error_message("ERROR_OPERATION_INPUTS_NOT_BROADCASTABLE")
                             (node.op.name, input_buf.name, "coeff", input_buf.shape, coeff_shape))

    def axes_to_spatial_first_order(self, node, graph):
        ret = super(OptimizePreluTranslation, self).axes_to_spatial_first_order(node, graph)
        if ret:
            # Input buffer axis might have been transformed, coeff need to be transformed as well
            OptimizePreluTranslation._permute_coeff(node, graph)
        return ret

    def merge_low_level_ops_to_layers(self, graph):
        def validate(node_tuple):
            first_node = node_tuple[0]
            if first_node.op.type == 'elementwise_min':
                min_node = first_node
                mul_node = node_tuple[1]
                max_node = node_tuple[2]
            else:
                max_node = first_node
                min_node = node_tuple[1]
                mul_node = node_tuple[2]

            min_input_buffer = graph.get_input_buffers(min_node)
            max_input_buffer = graph.get_input_buffers(max_node)
            mul_output_buffer = graph.get_output_buffers(mul_node)
            max_output_buffer = graph.get_output_buffers(max_node)

            if min_input_buffer[1] == max_input_buffer[1] and mul_output_buffer[0].consumers == max_output_buffer[0].consumers:
                return True
            return False

        sequence1 = [
            ("elementwise_min",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_max",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ()
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_max", "ANY"), ("elementwise_product", "ANY")]),
             ()
             ),
        ]
        sequence2 = [
            ("elementwise_max",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_min",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_max", "ANY"), ("elementwise_product", "ANY")]),
             ()
             ),
        ]
        sequences = [sequence1, sequence2]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate, ignore_constants=True)
            for node_tuple in matched_node_list:
                first_node = node_tuple[0]
                if first_node.op.type == 'elementwise_min':
                    mul_node = node_tuple[1]
                else:
                    mul_node = node_tuple[2]
                const_mul_node = graph.get_node_by_name(mul_node.input_names[0])
                add_node = node_tuple[3]

                # get the prelu coeff from the constant tensor and create a prelu op
                # Change the coeff to a constant node
                prelu_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.PreluOp.type,
                                                                        op_adapter.PreluOp.LEGACY_TRANSLATION_KEY)
                prelu_op = op_adapter.PreluOp(prelu_op_name)

                # replace the last op in seq with prelu_op
                graph.replace(add_node.op, prelu_op)
                # min_input_names = min_node.input_names

                # get the buffer of first node in the sequence
                first_node_buf = graph.get_buffer(first_node.input_names[1])

                # prune all the nodes in the sequence except the last one
                for node in node_tuple[:-1]:
                    graph.prune(node, force_remove=True)

                # update the input names of the prelu node
                prelu_node = graph.nodes_by_name[prelu_op.name]
                prelu_node.input_names = first_node.input_names[1:]
                prelu_node.input_names.append(const_mul_node.op.name)
                graph.get_buffer(const_mul_node.op.name).consumers.add(prelu_node)

                # update the consumers of the first node buffer
                first_node_buf.consumers.add(prelu_node)

    def prepare_inputs_as_params(self, node, graph):
        coeff_buffer = graph.get_buffer(node.input_names[1])
        coeff_node = coeff_buffer.producer
        if coeff_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.coeff = coeff_node.op.tensor
            # Remove the coeff inputs from the IR graph
            graph.remove_node_as_consumer(node, coeff_buffer.name)
            node.input_names = [node.input_names[0]]

@register_layer_optimization
class OptimizeProposalTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ProposalOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        output_buffer = graph.get_output_buffers(node)[0]

        # change input dims to 4D as required by snpe. Handling this here since converter allows for
        # non-4D inputs. Note: only change dimensions if it is input and no other node is consuming it
        # TODO: how should this be really handled
        im_info_input_buf = graph.get_input_buffers(node)[-1]
        if im_info_input_buf.producer.op.type == op_adapter.InputOp.TRANSLATION_KEY \
                and len(im_info_input_buf.consumers) == 1 \
                and im_info_input_buf.rank() != 4:
            shape = translation_utils.expand_to_rank(im_info_input_buf.shape, 4)
            im_info_input_buf.shape = shape
            im_info_input_buf.producer.op.shape = shape
            im_info_input_buf.axis_format = AxisTracker.AxisFormat.NSC
            output_buffer.axis_format = AxisTracker.AxisFormat.NSC
            return True
        else:
            return super(OptimizeProposalTranslation, self).axes_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeQuantizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.QuantizeOp.TRANSLATION_KEY
        self.register_method(REMOVE_QUANT_NODES, self.remove_quant_nodes)
        self.register_method(SQUASH_QUANT_NODES, self.squash_quant_dequant_to_convert)

    @staticmethod
    def squash_quant_dequant_to_convert(graph):
        sequence = [
                    (op_adapter.QuantizeOp.TRANSLATION_KEY,
                        (),
                        ()
                     ),
                    (op_adapter.DequantizeOp.TRANSLATION_KEY,
                        (),
                        ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence)
        for node_tuple in matched_node_list:

                # We found a quant/dequant combo, extract the nodes.
                first, second = node_tuple
                second_input_buffer = graph.get_input_buffers(second)[0]
                first_input_buffer = graph.get_input_buffers(first)[0]
                first_output_buffer = graph.get_output_buffers(first)[0]
                producer = first_input_buffer.producer

                # Fold these nodes into a convert op. Quant params are folded as part of squashing
                convert_name = producer.output_names[0] + "_convert_quant_dequant"
                convert_op = op_adapter.ConvertOp(convert_name)
                convert_node = graph.inject(convert_op, input_name=first_input_buffer.name, output_name=convert_name, consumer_names=[first.op.name])
                convert_input_buffer = graph.get_output_buffers(producer)[0]
                log_debug('Injecting convert op {} with input {} and output {}'.format(convert_name, convert_input_buffer.name, convert_name))
                convert_output_buffer = graph.get_output_buffers(convert_node)[0]
                log_debug('Found {} and {} nodes to squash into {} '.format(first.op.name,second.op.name,convert_op.name))
                graph.squash(second, input_name=second_input_buffer.name)
                graph.squash(first, input_name=convert_output_buffer.name)

    @staticmethod
    def remove_quant_nodes(node, graph):
        # Squash the quant node. The quant params are folded as part of squashing
        graph.squash(node, input_name=node.input_names[0])
        log_debug("Remove quantize op {}".format(node.op.name))


class OptimizeReduceTranslationBase(OptimizationTranslationBase):
    def __init__(self, op_type):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_type

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            target_format = spatial_first_format_to_channel_first_format[input_buf.axis_format]
            permute_order = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                if output_buf.axis_format != AxisTracker.AxisFormat.NC:
                    graph.inject_implicit_permute(input_name, target_format,
                                                  permute_order, [node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                else:
                    axis_map = permute_order
                    node.op.axes = [axis_map[axis] for axis in node.op.axes]
            else:
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]

        return True


@register_layer_optimization
class OptimizeReduceMaxTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX])


@register_layer_optimization
class OptimizeReduceMeanTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN])


@register_layer_optimization
class OptimizeReduceMinTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN])


@register_layer_optimization
class OptimizeReduceProdTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD])


@register_layer_optimization
class OptimizeReduceSumTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM])


@register_layer_optimization
class OptimizeReduceL2Translation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.IR_OP_REDUCE_L2])


@register_layer_optimization
class OptimizeReshapeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReshapeOp.TRANSLATION_KEY
        self.register_method(MATCH_CHANNELSHUFFLE, self.match_channelshuffle)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)
        self.register_method(SQUASH_RESHAPE, self.squash_reshape)
        self.register_method(FOLD_RESHAPES, self.fold_reshapes)
        self.register_method(ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE, self.add_transpose_after_output_reshape)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if AxisTracker.input_axis_formats_intact(graph, node) and \
                input_buf.axis_format in AxisTracker.AxisFormat.get_valid_formats():
            return False

        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.TransposeOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NC or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.TNF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

            return True

    @staticmethod
    def add_transpose_after_output_reshape(node, graph):
        """
        when output of graph is reshape, we need to add a transpoe after reshape so graph output axis format is consistent with graph input axis format
        e.g.,
                    op                                                         op (Channel_last)
                    | (Channel_first)                                          | (Channel_last)
                 reshape (Channel_first)      axes_to_spatial_first_orde   transpose (Channel_last)
                    | (Channel_first) (output)    ---------------->            | (Channel_first)
                                                                            reshape (Channel_first)
                                                                               | (Channel_first)
                                                                           transpose (Channel_first) <- should add this node so it has consistent axis format
                                                                               | (Channel_last) (output)
        """
        output_buf = graph.get_output_buffers(node)[0]
        input_node = graph.get_input_buffers(node)[0].producer
        axis_formats = [output_buf.axis_format]
        if len(output_buf.consumers) == 0:
            perm = None
            if output_buf.axis_format == AxisTracker.AxisFormat.NCF:
                perm = AxisTracker.AxisFormat.NCF_TO_NFC
            elif output_buf.axis_format == AxisTracker.AxisFormat.NCS:
                perm = AxisTracker.AxisFormat.NCS_TO_NSC
            elif output_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                perm = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            elif output_buf.axis_format == AxisTracker.AxisFormat.TNF:
                perm = AxisTracker.AxisFormat.TNF_TO_NTF
            if perm is not None:
                transpose_op_name = node.op.name + '_transpose'
                transpose_op = op_adapter.TransposeOp(transpose_op_name, perm=perm)
                post_reshape_idx = graph.nodes_in_order.index(node)
                # first prune and add back to adjust its output name
                graph.prune(node)
                post_reshape_output_name = node.output_names[0]
                new_post_reshape_output_name = input_node.output_names[0] + "." + output_buf.axis_format.lower()
                post_reshape_op = node.op
                node = graph.add(post_reshape_op, input_node.output_names, [new_post_reshape_output_name], idx=post_reshape_idx, axis_formats=axis_formats)
                graph.add(transpose_op, node.output_names, [post_reshape_output_name], idx=post_reshape_idx+1)

    @staticmethod
    def match_channelshuffle(graph):
        def is_valid_channelshuffle(nodes_tuple):
            def check_for_valid_reshape_1(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_1_input_shape = input_buffer.shape
                reshape_1_output_shape = output_buffer.shape

                return (len(reshape_1_input_shape) == 4 and len(reshape_1_output_shape) == 5 and
                        reshape_1_input_shape[0] == reshape_1_output_shape[0] and
                        reshape_1_input_shape[2] == reshape_1_output_shape[3] and
                        reshape_1_input_shape[3] == reshape_1_output_shape[4])

            def check_for_valid_permute(node):
                # Assuming the input shape is N[GC']HW
                return node.op.type == op_adapter.TransposeOp.TRANSLATION_KEY and node.op.perm == [0, 2, 1, 3, 4]

            def check_for_valid_reshape_2(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_2_input_shape = input_buffer.shape
                reshape_2_output_shape = output_buffer.shape

                return (len(reshape_2_input_shape) == 5 and len(reshape_2_output_shape) == 4 and
                        reshape_2_input_shape[0] == reshape_2_output_shape[0] and
                        reshape_2_input_shape[3] == reshape_2_output_shape[2] and
                        reshape_2_input_shape[4] == reshape_2_output_shape[3])

            first_, second_, third_ = nodes_tuple
            input_shape_ = graph.get_input_buffers(first_)[0].shape
            output_shape_ = graph.get_output_buffers(third_)[0].shape

            return ((output_shape_ == input_shape_) and
                    check_for_valid_reshape_1(first_) and
                    check_for_valid_permute(second_) and
                    check_for_valid_reshape_2(third_))

        sequence = [
                    ("Reshape",
                        (),
                        ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
                     ),
                    ("Transpose",
                        (),
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
                     ),
                    ("Reshape",
                        (),
                        ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_channelshuffle, ignore_constants=True)

        for node_tuple in matched_node_list:
            #  ChannelShuffle Op found, Squash Permute and 2nd Reshape Op and Replace 1st ReshapeOp with ShuffleOp
            first, second, third = node_tuple
            output_shape = graph.get_output_shapes(first)[0]
            # Assuming the shape is N[GC']HW
            groups = output_shape[1]

            third_input_buffer = graph.get_input_buffers(third)[0]
            graph.squash(third, input_name=third_input_buffer.name)

            second_input_buffer = graph.get_input_buffers(second)[0]
            graph.squash(second, input_name=second_input_buffer.name)

            shuffle_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.ChannelShuffleOp.type,
                                                                      op_adapter.ChannelShuffleOp.LEGACY_TRANSLATION_KEY)
            shuffle_op = op_adapter.ChannelShuffleOp(shuffle_op_name, num_groups=groups)

            graph.replace(first.op, shuffle_op)
            log_debug2(code_to_message.get_debugging_message("DEBUG_CHANNEL_SHUFFLE_REPLACE")(first.op.name,
                                                                                              second.op.name,
                                                                                              third.op.name,
                                                                                              shuffle_op.name))

    @staticmethod
    def remove_identity(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        consumers = list(graph.get_buffer(node.output_names[0]).consumers)
        ret = False
        # Remove reshape if same shape as input as this reshape has no effect, remove it
        if input_buffer.shape == output_buffer.shape and len(input_buffer.consumers) == 1:
            try:
                graph.squash_identity(node)
                log_debug("Squash Reshape op {} due to identity. "
                          "Input shape {}, shape after {}".format(node.op.name,
                                                                  input_buffer.shape,
                                                                  output_buffer.shape))
            except RuntimeError as e:
                log_debug("Squash Reshape op {} due to identity not possible ".format(node.op.name))
        # Remove reshape  if the batch dimension is maintained through the reshape when consumer of reshape is
        # fc layer
        elif len(consumers) == 1 and isinstance(consumers[0].op, op_adapter.FullyConnectedOp) and \
                 input_buffer.shape[0] == output_buffer.shape[0]:
            try:
                graph.squash(node, input_name=input_buffer.name, squash_into_next=True)
                log_debug("Squash Reshape op {} due to identity. "
                          "Input shape {}, shape after {}".format(node.op.name,
                                                                  input_buffer.shape,
                                                                  output_buffer.shape))
            except RuntimeError as e:
                log_debug("Squash Reshape op {} due to identity not possible ".format(node.op.name))

    @staticmethod
    def squash_reshape(graph):
        def validate_node(nodes_tuple):
            input_buffer = graph.get_buffer(nodes_tuple[0].input_names[0])
            return len(input_buffer.consumers) == 1

        def squash_reshape_into_constant(graph, node):
            constant_buffer = graph.get_buffer(node.input_names[0])

            const_tensor = constant_buffer.producer.op.tensor
            const_tensor_shape = graph.get_output_shapes(node)[0]
            const_tensor = np.reshape(const_tensor, const_tensor_shape)

            constant_buffer.producer.op.tensor = const_tensor
            constant_buffer.shape = const_tensor_shape

            log_debug("Squashed {} node {} into constant node {}"
                       .format(node.op.type, node.op.name, constant_buffer.name))
            graph.squash(node, input_name=constant_buffer.name)

        sequence = [
            ("Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            squash_reshape_into_constant(graph, node_tuple[0])

    @staticmethod
    def fold_reshapes(graph):
        def validate_node(nodes_tuple):
            input_buffer = graph.get_buffer(nodes_tuple[0].input_names[0])
            return len(input_buffer.consumers) == 1

        sequence = [
                    ("Reshape",
                     ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                     ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            reshape_node = node_tuple[0]
            reshape_node_input_buf = graph.get_input_buffers(reshape_node)[0]
            reshape_node_input_names = reshape_node.input_names

            if reshape_node_input_buf.producer.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                prev_reshape_node = reshape_node_input_buf.producer
                prev_reshape_node_input_names = prev_reshape_node.input_names

                # squash next reshape node into previous
                graph.squash(reshape_node, reshape_node_input_names[0], squash_into_next=False)

                log_debug2("Folded Reshape:{} into Reshape:{}".format(reshape_node.op.name, prev_reshape_node.op.name))


@register_layer_optimization
class OptimizeResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ResizeOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        return True


@register_layer_optimization
class OptimizeRoiAlignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == node.op.data_axis_formats[0] and \
                input_buf.axis_format != AxisTracker.AxisFormat.NCS:
            # No change
            return False

        AxisTracker.enforce_input_axis_format(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                              AxisTracker.AxisFormat.NCS_TO_NSC, valid_input_axis_formats=[AxisTracker.AxisFormat.NCS],
                                              consumers=[node.op.name])
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC
        return True


@register_layer_optimization
class OptimizeRoiPoolingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiPoolingOp.TRANSLATION_KEY
        self.register_method("PREPROCESS_ROI_POOL_INPUTS", self.preprocess_roi_pool_inputs)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == node.op.data_axis_formats[0]:
            # No change
            return False

        AxisTracker.enforce_input_axis_format(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                              AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC
        return True

    @staticmethod
    def preprocess_roi_pool_inputs(graph):
        def validate_node(nodes_tuple):
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])
            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY or len(roi_node.input_names) == 3:
                return True
            return False

        sequence = [(op_adapter.RoiPoolingOp.TRANSLATION_KEY, (), ())]

        matched_nodes_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for nodes_tuple in matched_nodes_list:
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])

            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                if roi_buf.shape[-1] == 5:
                    # QNN needs roi values to be separated from batch index
                    roi_values = roi_buf.producer.op.tensor
                    roi_values_no_batch = roi_values[:, 1:]

                    # Update ROI values in constant op to new values
                    roi_buf.producer.op.tensor = roi_values_no_batch

                    # Set batch indices to first sub-tensor of ROI values
                    batch_indices_name = roi_buf.name + "_batch_indices"
                    batch_indices = np.asarray(roi_values[:, 0], np.int32)

                    # Add a new constant op to capture batch indices

                    # constant op needs to be added before roi node
                    roi_idx = graph.nodes_in_order.index(roi_node)
                    graph.add(op_adapter.ConstantOp(batch_indices_name, batch_indices, quantizable=False), [],
                              [batch_indices_name], idx=roi_idx)

                    # add input name to roi node
                    roi_node.input_names.append(batch_indices_name)

                else:
                    raise ValueError("Expected 5 dimensions for static ROI buffer: {}, instead got {}"
                                     .format(roi_buf.name, roi_buf.shape[-1]))
            elif len(roi_node.input_names) != 3:
                raise AttributeError("Missing batch indices input. "
                                     "Expected 3 inputs for ROI operation instead got: {}"
                                     .format(len(roi_node.input_names)))


@register_layer_optimization
class OptimizeRolledLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RolledLstmOp.TRANSLATION_KEY
        self.register_method(UNROLL_LSTM_TIME_STEPS, self.unroll_lstm_time_steps)

    def axes_to_spatial_first_order(self, node, graph):
        # LSTM input axis format must be NTF
        input_bufs = graph.get_input_buffers(node)
        output_bufs = graph.get_output_buffers(node)

        # Enforce Data Buffer is NTF
        in_buf = input_bufs[0]
        if in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                in_buf.axis_format == AxisTracker.AxisFormat.TNF:
            graph.inject_implicit_permute(in_buf.name, AxisTracker.AxisFormat.NTF,
                                          AxisTracker.AxisFormat.TNF_TO_NTF)

        # Check that h/c input buffers are NONTRIVIAL
        for i, in_buf in enumerate(input_bufs[1:3]):
            if in_buf.type != op_graph.BufferType.NULL:
                log_assert(in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL,
                           "LSTM h/c input buffer {} needs to have format NONTRIVIAL, got {}",
                           in_buf,
                           in_buf.axis_format)

        # Set up LSTM outputs' axis formats
        # First output: NTF
        # Other outputs: NONTRIVIAL
        for i, output_buf in enumerate(output_bufs):
            if i == 0:
                output_buf.axis_format = AxisTracker.AxisFormat.NTF
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        return True

    @staticmethod
    def unroll_lstm_time_steps(graph):
        # Index for QNN LstmOp inputs and outputs
        DATA_IDX, HIDDEN_IN_IDX, CELL_IN_IDX = 0, 10, 11
        # LstmOp is defined as single timestep in QNN, so output tensor
        # at index HIDDEN_OUT_IDX and HIDDEN_ALL_OUT_IDX are effectively the same
        HIDDEN_ALL_OUT_IDX, CELL_OUT_IDX, HIDDEN_OUT_IDX = 0, 1, 2

        def squash_input_reshape_if_possible(rolled_lstm_node):
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[DATA_IDX]).shape
            batch_size, _, input_size = input_shape
            candidate_reshape_node = graph.get_producer_node(rolled_lstm_node.input_names[DATA_IDX])
            if candidate_reshape_node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                input_buffers = graph.get_input_buffers(candidate_reshape_node)
                output_buffers = graph.get_output_buffers(candidate_reshape_node)
                # check if no other consumers and input shape for this buffer is 2D i.e this is an unsqueeze
                # remove reshape to revert back to 2D, input_reshape is not needed
                # otherwise reshape is needed
                if len(output_buffers[0].consumers) == 1 and input_buffers[0].shape == [batch_size, input_size]:
                    if graph.squash(candidate_reshape_node, input_buffers[0].name):
                        # update lstm node input buffer shape to pre-reshape input shape
                        rolled_lstm_node_input_buffer = graph.get_buffer(rolled_lstm_node.input_names[0])
                        rolled_lstm_node_input_buffer.shape = input_buffers[0].shape
                        # if reshape node is squashed, return True
                        return True
            return False

        def split_input(rolled_lstm_node, time_step_axis=1):
            rolled_lstm_node_name = rolled_lstm_node.op.name
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[DATA_IDX]).shape
            seq_length = input_shape[1]
            input_x_split_name_list = []

            for i in range(seq_length):
                input_x_i_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[DATA_IDX] + str(i)
                input_x_split_name_list.append(input_x_i_name)
            input_x_split_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[DATA_IDX] + "_split"
            # If split_index is not specified, we split equally between the number of outputs
            input_x_split_op = op_adapter.SplitOp(name=input_x_split_name, axis=time_step_axis)

            # The split T inputs have same rank as original input, so reshape is needed to squeeze the timestep dimension
            graph.add(input_x_split_op, input_names=[rolled_lstm_node.input_names[DATA_IDX]],
                      output_names=input_x_split_name_list, idx=rolled_lstm_node_idx)
            return input_x_split_name_list

        def reshape_input(rolled_lstm_node, input_x_name_list):
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[DATA_IDX]).shape
            batch_size, _, input_size = input_shape
            for i, input_x_name in enumerate(input_x_name_list):
                input_x_reshape_name = input_x_name + "_reshape"
                # Bidirectional lstm share the same input X, so add a check here
                if not graph.has_buffer(input_x_reshape_name):
                    input_x_reshape_output_shape = [batch_size, input_size]
                    input_x_reshape_op = op_adapter.ReshapeOp(name=input_x_reshape_name,
                                                              shape=input_x_reshape_output_shape)
                    graph.add(input_x_reshape_op, input_names=[input_x_name],
                              output_names=[input_x_reshape_name], idx=rolled_lstm_node_idx)
                    # Update the RolledLstm index for adding a reshape to the graph
                    rolled_lstm_node_idx += 1

                input_x_name_list[i] = input_x_reshape_name

        def prepare_lstm_output_name_list(rolled_lstm_node):
            output_y_name_list = []
            output_y_reshape_name_list = []
            output_h_name_list = []
            output_c_name_list = []
            seq_length = graph.get_buffer(rolled_lstm_node.input_names[DATA_IDX]).shape[1]
            for i in range(seq_length):
                output_y_i_name = rolled_lstm_node.output_names[HIDDEN_ALL_OUT_IDX] + str(i)
                output_y_name_list.append(output_y_i_name)
                output_y_reshape_i_name = output_y_i_name + "_reshape"
                output_y_reshape_name_list.append(output_y_reshape_i_name)
                output_h_i_name = rolled_lstm_node.output_names[HIDDEN_OUT_IDX] + str(i)
                output_h_name_list.append(output_h_i_name)
                output_c_i_name = rolled_lstm_node.output_names[CELL_OUT_IDX] + str(i)
                output_c_name_list.append(output_c_i_name)

            return output_y_name_list, output_y_reshape_name_list, output_h_name_list, output_c_name_list

        def add_single_timestep_lstm_op(rolled_lstm_node, reset_state_at_time_step_0, h_0_input_name, c_0_input_name,
                                        lstm_time_step_i_op_name, lstm_i_node_input_name_list, lstm_i_node_output_name_list):
            lstm_time_step_i_op = op_adapter.LstmOp(name=lstm_time_step_i_op_name,
                                                    hidden_size=rolled_lstm_node.op.hidden_size,
                                                    direction=rolled_lstm_node.op.direction,
                                                    reset_state_at_time_step_0=reset_state_at_time_step_0,
                                                    h_0_input_name=h_0_input_name,
                                                    c_0_input_name=c_0_input_name,
                                                    sequence_continuation_name=rolled_lstm_node.op.sequence_continuation_name,
                                                    x_static_name=rolled_lstm_node.op.x_static_name,
                                                    cell_clip_threshold=rolled_lstm_node.op.cell_clip_threshold,
                                                    output_clip_threshold=rolled_lstm_node.op.output_clip_threshold)
            graph.add(lstm_time_step_i_op,
                      input_names=lstm_i_node_input_name_list,
                      output_names=lstm_i_node_output_name_list,
                      idx=graph.nodes_in_order.index(rolled_lstm_node))

        def add_lstm_output_reshape(rolled_lstm_node, output_name, output_reshape_name):
            batch_size = graph.get_buffer(rolled_lstm_node.input_names[DATA_IDX]).shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[HIDDEN_ALL_OUT_IDX]).shape[-1]

            output_all_h_reshape_output_shape = [batch_size, 1, output_size]
            output_all_h_reshape_op = op_adapter.ReshapeOp(name=output_reshape_name,
                                                           shape=output_all_h_reshape_output_shape)
            graph.inject(output_all_h_reshape_op,
                         input_name=output_name,
                         output_name=output_reshape_name,
                         consumer_names=[consumer.op.name for consumer in list(graph.get_buffer(output_name).consumers)])

            # Setting up reshape output buffer axis format to be NTF
            graph.get_buffer(output_reshape_name).axis_format = AxisTracker.AxisFormat.NTF
            # Change output buffer shape to 2D
            graph.get_buffer(output_name).shape = [batch_size, output_size]
            graph.get_buffer(output_name).axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        def concat_multi_timestep_outputs(rolled_lstm_node, concat_input_name_list, concat_output_name, time_step_axis=1):
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            output_y_concat_op = op_adapter.ConcatOp(name=concat_output_name, axis=time_step_axis)
            if rolled_lstm_node.op.direction == ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE:
                concat_input_name_list.reverse()
            graph.add(output_y_concat_op, input_names=concat_input_name_list, output_names=[concat_output_name], idx=rolled_lstm_node_idx)

        def adjust_lstm_output_consumers(rolled_lstm_node, output_all_hidden_concat_name, output_hidden_name, output_cell_name):
            # Extract output buffers
            all_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[HIDDEN_ALL_OUT_IDX])
            hidden_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[HIDDEN_OUT_IDX])
            cell_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[CELL_OUT_IDX])

            # Add the original output buffer consumers to the last lstm_i node
            for consumer in list(all_output_buffer.consumers):
                output_all_h_concat_buffer = graph.get_buffer(output_all_hidden_concat_name)
                output_all_h_concat_buffer.consumers.add(consumer)
                h_all_idx = consumer.input_names.index(rolled_lstm_node.output_names[HIDDEN_ALL_OUT_IDX]) + 1
                consumer.input_names.insert(h_all_idx, output_all_hidden_concat_name)

            for consumer in list(hidden_output_buffer.consumers):
                output_h_buffer = graph.get_buffer(output_hidden_name)
                output_h_buffer.consumers.add(consumer)
                h_idx = consumer.input_names.index(rolled_lstm_node.output_names[HIDDEN_OUT_IDX]) + 1
                consumer.input_names.insert(h_idx, output_hidden_name)

            for consumer in list(cell_output_buffer.consumers):
                ourput_c_buffer = graph.get_buffer(output_cell_name)
                ourput_c_buffer.consumers.add(consumer)
                c_idx = consumer.input_names.index(rolled_lstm_node.output_names[CELL_OUT_IDX]) + 1
                consumer.input_names.insert(c_idx, output_cell_name)

        def align_to_source_output_names(current_output_names, source_output_names):
            # Replace current name with source name for alignment
            for current_name, source_name in zip(current_output_names, source_output_names):
                buf = graph.get_buffer(current_name)
                if source_name in graph.buffers:
                    raise ValueError("Buffer {} already exists in graph, duplicate buffer name when replacing buffer {} with it".format(
                            source_name, current_name))

                # Update consumers input name
                for consumer in list(buf.consumers):
                    # The consumer may have the same buffer as input twice
                    consumer.input_names = [source_name if name == current_name else name for name in consumer.input_names]

                # Update producer output name
                producer_node = graph.get_producer_node(current_name)
                idx = producer_node.output_names.index(current_name)
                producer_node.output_names[idx] = source_name

                # Update buffer in graph
                buf.name = source_name
                graph.buffers[source_name] = graph.buffers.pop(current_name)

        sequence = [
            (op_adapter.RolledLstmOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            rolled_lstm_node = nodes_tuple[0]
            rolled_lstm_node_name = rolled_lstm_node.op.name
            log_debug("Unrolling RolledLstm node {}".format(rolled_lstm_node_name))

            # Extract and validate sizes
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[DATA_IDX]).shape
            seq_length = input_shape[1]
            time_step_axis = 1
            if len(input_shape) != 3:
                raise ValueError('Unsupported input rank for RolledLstm node {}, expected 3, got {}.'.format(
                     rolled_lstm_node_name, len(input_shape)))

            # The name list of 2D input X_i for lstm_i node(s) at timestep i
            input_x_name_list = []
            # This variable determines if a reshape needs to be added to the input
            input_reshape_needed = True

            if seq_length == 1:
                # Since sequence length is one, we need to squeeze the dimension to 2D
                # We can do this by removing a reshape which may have been added by the frontend
                # Or by adding a reshape ourselves to squeeze to 2D.
                if squash_input_reshape_if_possible(rolled_lstm_node):
                    # set no reshape needed
                    input_reshape_needed = False
                # Add the name to the input name list
                input_x_name_list.append(rolled_lstm_node.input_names[DATA_IDX])
            else:
                input_x_split_name_list = split_input(rolled_lstm_node, time_step_axis=time_step_axis)
                # Add the input x split names to input name list
                input_x_name_list.extend(input_x_split_name_list)

            # Adding reshape nodes to squeeze sequence length dimensions from input if input is 3D
            if input_reshape_needed:
                reshape_input(rolled_lstm_node, input_x_name_list)

            # Pre-process RolledLstm node and return the input name list for LstmOp
            lstm_all_inputs_name_list = OptimizeRolledLstmTranslation.preprocess_rolled_lstm_node(graph, rolled_lstm_node)
            output_y_name_list, output_y_reshape_name_list, output_h_name_list, output_c_name_list = prepare_lstm_output_name_list(rolled_lstm_node)

            # Add LstmOp to the graph per timestep
            for i in range(seq_length):
                # Prepare name of LstmOp at timestep i
                lstm_time_step_i_op_name = rolled_lstm_node_name + '_step_' + str(i)
                # Prepare necessary attributes for lstm_i
                reset_state_at_time_step_0 = rolled_lstm_node.op.reset_state_at_time_step_0 if i == 0 else False
                h_0_input_name = rolled_lstm_node.op.h_0_input_name if i == 0 else output_h_name_list[i-1]
                c_0_input_name = rolled_lstm_node.op.c_0_input_name if i == 0 else output_c_name_list[i-1]

                # Share weights and biases across Lstm nodes by using the same input name list
                lstm_i_node_input_name_list = lstm_all_inputs_name_list[:]
                # Update the specific inputs for lstm_i
                curr_idx = i if rolled_lstm_node.op.direction == ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD else seq_length-1-i
                lstm_i_node_input_name_list[DATA_IDX] = input_x_name_list[curr_idx]
                lstm_i_node_input_name_list[HIDDEN_IN_IDX] = h_0_input_name
                lstm_i_node_input_name_list[CELL_IN_IDX] = c_0_input_name
                # Prepare output name list for lstm_i
                lstm_i_node_output_name_list = [output_y_name_list[i], output_c_name_list[i], output_h_name_list[i]]

                add_single_timestep_lstm_op(rolled_lstm_node, reset_state_at_time_step_0, h_0_input_name, c_0_input_name,
                                            lstm_time_step_i_op_name, lstm_i_node_input_name_list, lstm_i_node_output_name_list)
                # Reshape is added to unsqueeze the timestep dimension if output buffer is not 2D and
                # it is necessary to restore the NTF axis format regarding the output shape of 2D from QNN LstmOp
                add_lstm_output_reshape(rolled_lstm_node, lstm_i_node_output_name_list[HIDDEN_ALL_OUT_IDX], output_y_reshape_name_list[i])

            output_y_concat_name = rolled_lstm_node.output_names[HIDDEN_ALL_OUT_IDX] + "_reshape_concat" if seq_length > 1 else \
                                   output_y_reshape_name_list[0]
            # Concat output from T outputs
            if seq_length > 1:
                concat_multi_timestep_outputs(rolled_lstm_node, output_y_reshape_name_list, output_y_concat_name, time_step_axis=time_step_axis)

            adjust_lstm_output_consumers(rolled_lstm_node, output_y_concat_name, output_h_name_list[seq_length-1], output_c_name_list[seq_length-1])
            source_output_names = rolled_lstm_node.output_names

            # Prune original RolledLstm node
            graph.prune(rolled_lstm_node, force_remove=True)

            current_output_names = [output_y_concat_name, output_c_name_list[seq_length-1], output_h_name_list[seq_length-1]]
            # At this point, current output names are not aligned to source output names, we need to
            # restore the source output names from RolledLstm node
            align_to_source_output_names(current_output_names, source_output_names)

    # TODO Move to QNN-specific graph transformations once work on GraphTransformer is complete
    # Preprocesses LstmOp inputs, outputs, and attributes for QNN consumption
    @staticmethod
    def preprocess_rolled_lstm_node(graph, rolled_lstm_node):
        # Index for RolledLstmOp inputs
        INPUT_WEIGHTS_IDX, HIDDEN_STATE_WEIGHTS_IDX, GATE_BIASES_IDX = 3, 4, 5
        NORM_WEIGHTS_IDX, CELL_STATE_WEIGHTS_IDX, PROJ_WEIGHTS_IDX, PROJ_BIAS_IDX = 6, 7, 8, 9

        def split_lstm_tensor_per_gate(input_name, split_axis=0):
            producer_node = graph.get_producer_node(input_name)
            if producer_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                param_tensor = producer_node.op.tensor
                # Split weights so that they can be indexed by gate
                split_sections = int(param_tensor.shape[split_axis] / rolled_lstm_node.op.hidden_size)
                param_split_tensor = np.split(param_tensor, indices_or_sections=split_sections, axis=split_axis)
                # Two different RolledLstmOps may share the same weights or biases, so we need to extract the
                # weights by input name before we prune the const node from graph
                param_buf_consumers = graph.get_buffer(input_name).consumers
                param_buf_consumers.remove(rolled_lstm_node)
                if not param_buf_consumers:
                    # Prune the unsplit weights node from the graph
                    graph.prune(producer_node, force_remove=True)
                return param_split_tensor
            else:
                raise ValueError("LstmOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                        producer_node.op.name))

        def add_split_tensor_to_graph(tensor_name, tensor, desired_shape=None):
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            # Share the tensor if they are already added in the graph
            if not graph.has_buffer(tensor_name):
                tensor = np.resize(tensor, desired_shape) if desired_shape else tensor
                const_op = op_adapter.ConstantOp(name=tensor_name, tensor=tensor)
                graph.add(const_op, input_names=[], output_names=[tensor_name], idx=rolled_lstm_node_idx)
            elif graph.get_producer_op(tensor_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("LstmOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                        graph.get_producer_op(tensor_name).name))

        # Must add all inputs derived from splitting the tensor per gate as ConstantOp to the graph.
        # Weights may already be 2D (from TF as an example), but it is cleaner to resize anyway
        # rather than check shape for each input.
        # The weights and biases are shared across unrolled lstm nodes
        def prepare_lstm_all_inputs():
            input_size = graph.get_buffer(rolled_lstm_node.input_names[0]).shape[-1]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            # Input weights are expected in [4*hidden_size, input_size] in IFOC format
            src_input_weights_name = rolled_lstm_node.input_names[INPUT_WEIGHTS_IDX]
            input_split_weights = split_lstm_tensor_per_gate(src_input_weights_name)

            input_w_to_forget_gate_name = src_input_weights_name + '_input_w_to_forget_gate'
            add_split_tensor_to_graph(input_w_to_forget_gate_name, input_split_weights[1], desired_shape=(num_units, input_size))

            input_w_to_cell_gate_name = src_input_weights_name + '_input_w_to_cell_gate'
            add_split_tensor_to_graph(input_w_to_cell_gate_name, input_split_weights[3], desired_shape=(num_units, input_size))

            input_w_to_output_gate_name = src_input_weights_name + '_input_w_to_output_gate'
            add_split_tensor_to_graph(input_w_to_output_gate_name, input_split_weights[2], desired_shape=(num_units, input_size))

            # Hidden state weights are expected in [4*hidden_size, hidden_size] in IFOC format
            src_hidden_state_weights_name = rolled_lstm_node.input_names[HIDDEN_STATE_WEIGHTS_IDX]
            hidden_state_split_weights = split_lstm_tensor_per_gate(src_hidden_state_weights_name)

            recurrent_w_to_forget_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_forget_gate'
            add_split_tensor_to_graph(recurrent_w_to_forget_gate_name, hidden_state_split_weights[1], desired_shape=(num_units, output_size))

            recurrent_w_to_cell_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_cell_gate'
            add_split_tensor_to_graph(recurrent_w_to_cell_gate_name, hidden_state_split_weights[3], desired_shape=(num_units, output_size))

            recurrent_w_to_output_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_output_gate'
            add_split_tensor_to_graph(recurrent_w_to_output_gate_name, hidden_state_split_weights[2], desired_shape=(num_units, output_size))

            # Gate biases are expected in [4*hidden_size] in IFOC format
            src_gate_biases_name = rolled_lstm_node.input_names[GATE_BIASES_IDX]
            gate_split_biases = split_lstm_tensor_per_gate(src_gate_biases_name)

            b_to_forget_gate_name = src_gate_biases_name + '_b_to_forget_gate'
            add_split_tensor_to_graph(b_to_forget_gate_name, gate_split_biases[1], desired_shape=(num_units,))

            b_to_cell_gate_name = src_gate_biases_name + '_b_to_cell_gate'
            add_split_tensor_to_graph(b_to_cell_gate_name, gate_split_biases[3], desired_shape=(num_units,))

            b_to_output_gate_name = src_gate_biases_name + '_b_to_output_gate'
            add_split_tensor_to_graph(b_to_output_gate_name, gate_split_biases[2], desired_shape=(num_units,))

            # Normalization weights are expected in [4*hidden_size] in IFOC format
            src_norm_weights_name = rolled_lstm_node.input_names[NORM_WEIGHTS_IDX]
            norm_split_weights = split_lstm_tensor_per_gate(src_norm_weights_name) if src_norm_weights_name else None

            norm_w_to_input_gate_name = src_norm_weights_name + '_norm_w_to_input_gate' if norm_split_weights else ''
            if norm_w_to_input_gate_name:
                add_split_tensor_to_graph(norm_w_to_input_gate_name, norm_split_weights[0], desired_shape=(num_units,))

            norm_w_to_forget_gate_name = src_norm_weights_name + '_norm_w_to_forget_gate' if norm_split_weights else ''
            if norm_w_to_forget_gate_name:
                add_split_tensor_to_graph(norm_w_to_forget_gate_name, norm_split_weights[1], desired_shape=(num_units,))

            norm_w_to_cell_gate_name = src_norm_weights_name + '_norm_w_to_cell_gate' if norm_split_weights else ''
            if norm_w_to_cell_gate_name:
                add_split_tensor_to_graph(norm_w_to_cell_gate_name, norm_split_weights[3], desired_shape=(num_units,))

            norm_w_to_output_gate_name = src_norm_weights_name + '_norm_w_to_output_gate' if norm_split_weights else ''
            if norm_w_to_output_gate_name:
                add_split_tensor_to_graph(norm_w_to_output_gate_name, norm_split_weights[2], desired_shape=(num_units,))

            input_w_to_input_gate_name = src_input_weights_name + '_input_w_to_input_gate'
            add_split_tensor_to_graph(input_w_to_input_gate_name, input_split_weights[0], desired_shape=(num_units, input_size))

            recurrent_w_to_input_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_input_gate'
            add_split_tensor_to_graph(recurrent_w_to_input_gate_name, hidden_state_split_weights[0], desired_shape=(num_units, output_size))

            # Cell state weights are expected in [3*hidden_size] in IFO format
            src_cell_state_weights_name = rolled_lstm_node.input_names[CELL_STATE_WEIGHTS_IDX]
            cell_state_split_weights = split_lstm_tensor_per_gate(src_cell_state_weights_name) if src_cell_state_weights_name else None

            cell_w_to_input_gate_name = src_cell_state_weights_name + '_cell_w_to_input_gate' if cell_state_split_weights else ''
            if cell_w_to_input_gate_name:
                add_split_tensor_to_graph(cell_w_to_input_gate_name, cell_state_split_weights[0], desired_shape=(num_units,))

            cell_w_to_forget_gate_name = src_cell_state_weights_name + '_cell_w_to_forget_gate' if cell_state_split_weights else ''
            if cell_w_to_forget_gate_name:
                add_split_tensor_to_graph(cell_w_to_forget_gate_name, cell_state_split_weights[1], desired_shape=(num_units,))

            cell_w_to_output_gate_name = src_cell_state_weights_name + '_cell_w_to_output_gate' if cell_state_split_weights else ''
            if cell_w_to_output_gate_name:
                add_split_tensor_to_graph(cell_w_to_output_gate_name, cell_state_split_weights[2], desired_shape=(num_units,))

            b_to_input_gate_name = src_gate_biases_name + '_b_to_input_gate'
            add_split_tensor_to_graph(b_to_input_gate_name, gate_split_biases[0], desired_shape=(num_units,))

            # The projection weights and bias do not need to be split, and they are added to the graph in frontend if provided
            proj_w_name = rolled_lstm_node.input_names[PROJ_WEIGHTS_IDX]
            proj_b_name = rolled_lstm_node.input_names[PROJ_BIAS_IDX]

            # Prepare the LstmOp input names - inputs not captured by any FE are passed the empty string
            lstm_all_inputs_name_list = [
                rolled_lstm_node.input_names[0],
                input_w_to_forget_gate_name,
                input_w_to_cell_gate_name,
                input_w_to_output_gate_name,
                recurrent_w_to_forget_gate_name,
                recurrent_w_to_cell_gate_name,
                recurrent_w_to_output_gate_name,
                b_to_forget_gate_name,
                b_to_cell_gate_name,
                b_to_output_gate_name,
                rolled_lstm_node.input_names[1],
                rolled_lstm_node.input_names[2],
                norm_w_to_input_gate_name,
                norm_w_to_forget_gate_name,
                norm_w_to_cell_gate_name,
                norm_w_to_output_gate_name,
                input_w_to_input_gate_name,
                recurrent_w_to_input_gate_name,
                cell_w_to_input_gate_name,
                cell_w_to_forget_gate_name,
                cell_w_to_output_gate_name,
                b_to_input_gate_name,
                proj_w_name,
                proj_b_name
            ]

            # Update the RolledLstmOp input names
            rolled_lstm_node.input_names = rolled_lstm_node.input_names[:INPUT_WEIGHTS_IDX]
            return lstm_all_inputs_name_list

        def ensure_h_c_inputs_present():
                rolled_lstm_node_name = rolled_lstm_node.op.name
                rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
                batch_size = graph.get_buffer(rolled_lstm_node.input_names[0]).shape[0]
                output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
                num_units = rolled_lstm_node.op.hidden_size

                # Requires initial_h and initial_c inputs to be present
                # The following code adds zero valued tensors provided the conditions below are satisfied
                if not rolled_lstm_node.input_names[1] and not rolled_lstm_node.input_names[2]:
                    if rolled_lstm_node.op.h_0_input_name:
                        raise ValueError('RolledLstm node {} op attribute h_0_input_name {} mismatch with rolled_lstm_node.input_names[1] {}.'.format(
                                rolled_lstm_node_name, rolled_lstm_node.op.h_0_input_name, rolled_lstm_node.input_names[1]))
                    if rolled_lstm_node.op.c_0_input_name:
                        raise ValueError('RolledLstm node {} op attribute c_0_input_name {} mismatch with rolled_lstm_node.input_names[2] {}.'.format(
                                rolled_lstm_node_name, rolled_lstm_node.op.c_0_input_name, rolled_lstm_node.input_names[2]))

                    # add zeros for initial h and c inputs since there are needed for QNN
                    initial_hidden_state_name = rolled_lstm_node_name + '_initial_hidden_state'
                    initial_hidden_state_tensor = np.zeros((batch_size, output_size), dtype=np.float32)
                    initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor)
                    graph.add(initial_hidden_state_op, input_names=[], output_names=[initial_hidden_state_name], idx=rolled_lstm_node_idx)
                    rolled_lstm_node.input_names[1] = initial_hidden_state_name
                    rolled_lstm_node.op.h_0_input_name = initial_hidden_state_name
                    graph.get_buffer(initial_hidden_state_name).consumers.add(rolled_lstm_node)

                    initial_cell_state_name = rolled_lstm_node_name + '_initial_cell_state'
                    initial_cell_state_tensor = np.zeros((batch_size, num_units), dtype=np.float32)
                    initial_cell_state_op = op_adapter.ConstantOp(name=initial_cell_state_name, tensor=initial_cell_state_tensor)
                    graph.add(initial_cell_state_op, input_names=[], output_names=[initial_cell_state_name], idx=rolled_lstm_node_idx+1)
                    rolled_lstm_node.input_names[2] = initial_cell_state_name
                    rolled_lstm_node.op.c_0_input_name = initial_cell_state_name
                    graph.get_buffer(initial_cell_state_name).consumers.add(rolled_lstm_node)

        def add_h_c_inputs_reshape_if_needed():
            rolled_lstm_node_name = rolled_lstm_node.op.name
            batch_size = graph.get_buffer(rolled_lstm_node.input_names[0]).shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            # If the initial hidden state shape (and implicitly initial cell state shape)
            # is not 2D then it should be reshaped
            initial_h_shape = graph.get_buffer(rolled_lstm_node.input_names[1]).shape
            initial_state_reshape_needed = len(initial_h_shape) != 2 or initial_h_shape != [batch_size, output_size]
            if initial_state_reshape_needed:
                input_h_reshape_node_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[1] + "_reshape"
                input_h_reshape_output_shape = [batch_size, output_size]
                input_h_reshape_op = op_adapter.ReshapeOp(name=input_h_reshape_node_name,
                                                          shape=input_h_reshape_output_shape)
                graph.inject(input_h_reshape_op, input_name=rolled_lstm_node.input_names[1],
                             output_name=input_h_reshape_node_name, consumer_names=[rolled_lstm_node_name])
                rolled_lstm_node.op.h_0_input_name = input_h_reshape_node_name

                input_c_reshape_node_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[2] + "_reshape"
                input_c_reshape_output_shape = [batch_size, num_units]
                input_c_reshape_op = op_adapter.ReshapeOp(name=input_c_reshape_node_name,
                                                          shape=input_c_reshape_output_shape)
                graph.inject(input_c_reshape_op, input_name=rolled_lstm_node.input_names[2],
                             output_name=input_c_reshape_node_name, consumer_names=[rolled_lstm_node_name])
                rolled_lstm_node.op.c_0_input_name = input_c_reshape_node_name

        def handle_missing_outputs():
            rolled_lstm_node_name = rolled_lstm_node.op.name
            batch_size = graph.get_buffer(rolled_lstm_node.input_names[0]).shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            number_of_outputs = len(rolled_lstm_node.output_names)
            all_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[0])

            if number_of_outputs == 3:
                # Modify existing output buffers for QNN specification
                hidden_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[2])
                hidden_output_buffer.shape = [batch_size, output_size]
                hidden_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

                cell_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[1])
                cell_output_buffer.shape = [batch_size, num_units]
                cell_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

                # Prepare output names and keep the first output as all_hidden
                # Output tensor at index HIDDEN_OUT_IDX and HIDDEN_ALL_OUT_IDX are effectively the same in QNN LstmOp
                rolled_lstm_node.output_names = [all_output_buffer.name, cell_output_buffer.name, hidden_output_buffer.name]
            elif number_of_outputs == 1:
                # Add dummy buffers for missing outputs - QNN requires 3
                hidden_output_dummy_name = rolled_lstm_node_name + "_hidden_output_dummy"
                graph.add_output_buffer(rolled_lstm_node, hidden_output_dummy_name,
                                        [batch_size, output_size], AxisTracker.AxisFormat.NONTRIVIAL)

                cell_output_dummy_name = rolled_lstm_node_name + "_cell_output_dummy"
                graph.add_output_buffer(rolled_lstm_node, cell_output_dummy_name,
                                        [batch_size, num_units], AxisTracker.AxisFormat.NONTRIVIAL)

                rolled_lstm_node.output_names = [all_output_buffer.name, cell_output_dummy_name, hidden_output_dummy_name]
            else:
                # Only 1 or 3 outputs are supported for this optimization
                raise ValueError("Unsupported number of outputs for RolledLstm node {}, expected 1 or 3, got {}.".format(
                    rolled_lstm_node_name, number_of_outputs))

        log_debug("Preprocessing RolledLstm node {} for QNN lowering.".format(rolled_lstm_node.op.name))

        # Prepare QNN Lstm all inputs and return the input name list
        lstm_all_inputs_name_list = prepare_lstm_all_inputs()
        ensure_h_c_inputs_present()
        add_h_c_inputs_reshape_if_needed()
        handle_missing_outputs()

        return lstm_all_inputs_name_list

@register_layer_optimization
class OptimizeLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LstmOp.TRANSLATION_KEY
        self.register_method(EXPAND_LSTM_OP_STRUCTURE, self.expand_lstm_op_structure)


    # it = f(Xt*(Wxi^T) + Ht_1*(Whi^T) + Ct_1*(Wci^T) + Bi) if Bi and Wci are both present
    # it = f(Xt*(Wxi^T) + Ht_1*(Whi^T) + Bi) if Bi is present but Wci is absent
    # it = f(Ct_1*(Wci^T)) if Bi is absent but Wci is present
    # it = 1 - ft if Bi and Wci are both absent
    def expand_lstm_input_gate(self, graph, lstm_node, Xt, Ht_1, Ct_1, ft):
        lstm_node_name = lstm_node.op.name
        lstm_node_idx = graph.nodes_in_order.index(lstm_node)

        Wxi_idx, Whi_idx, Wci_idx, Bi_idx = 16, 17, 18, 21

        Wxi_name = lstm_node.input_names[Wxi_idx]
        Whi_name = lstm_node.input_names[Whi_idx]
        Wci_name = lstm_node.input_names[Wci_idx]
        Bi_name = lstm_node.input_names[Bi_idx]

        use_peephole_optimization = (Wci_name != '')
        coupled_input_forget = ((Wxi_name == '') and (Whi_name == '') and (Bi_name == ''))

        if coupled_input_forget :
            initial_ones_op_name = lstm_node_name + '_initial_ones_op'
            initial_ones_tensor = np.ones(graph.get_buffer(ft).shape, dtype=np.float32)
            initial_ones_op = op_adapter.ConstantOp(name=initial_ones_op_name, tensor=initial_ones_tensor)
            graph.add(initial_ones_op, input_names=[], output_names=[initial_ones_op_name], idx=graph.nodes_in_order.index(lstm_node)-1)

            one_minus_ft_op_name = lstm_node_name + "_one_minus_ft_op"
            one_minus_ft_op = op_adapter.ElementwiseBinaryOp(name=one_minus_ft_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT)
            graph.add(one_minus_ft_op, input_names=[initial_ones_op_name, ft],
                                output_names=[one_minus_ft_op_name], idx=graph.nodes_in_order.index(lstm_node))

            return one_minus_ft_op_name

        inputs_to_activation_input_op = []

        Bi_buf = graph.get_buffer(Bi_name)
        Bi_size = Bi_buf.shape[-1]
        xt_wxi_matmul_op_name = lstm_node_name + "_xt_wxi_matmul_op"
        Wbx =  np.zeros(Bi_size, dtype=np.float32)
        xt_wxi_matmul_op = op_adapter.MatMulOp(name=xt_wxi_matmul_op_name,
                                                bias=Wbx,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(xt_wxi_matmul_op, input_names=[Xt, Wxi_name],
                  output_names=[xt_wxi_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        ht_1_whi_matmul_op_name = lstm_node_name + "_ht_1_whi_matmul_op"
        Wbh =  np.zeros(Bi_size, dtype=np.float32)
        ht_1_whi_matmul_op = op_adapter.MatMulOp(name=ht_1_whi_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(ht_1_whi_matmul_op, input_names=[Ht_1, Whi_name],
                  output_names=[ht_1_whi_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_of_term1_part1_op_name = lstm_node_name + "_elementsum_of_term1_part1_op"
        elementsum_of_term1_part1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_part1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term1_part1_op, input_names=[xt_wxi_matmul_op_name, ht_1_whi_matmul_op_name],
                            output_names=[elementsum_of_term1_part1_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_of_term1_part1_bias_op_name = lstm_node_name + "_elementsum_of_term1_part1_bias_op"
        elementsum_of_term1_part1_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_part1_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term1_part1_bias_op, input_names=[elementsum_of_term1_part1_op_name, Bi_name],
                            output_names=[elementsum_of_term1_part1_bias_op_name], idx=graph.nodes_in_order.index(lstm_node))

        if use_peephole_optimization :
            ct_1_wci_matmul_op_name = lstm_node_name + "_ct_1_wci_matmul_op"
            Wbc =  np.zeros(graph.get_buffer(Wci_name).shape[-1], dtype=np.float32)
            ct_1_wci_matmul_op = op_adapter.MatMulOp(name=ct_1_wci_matmul_op_name,
                                                     bias=Wbc,
                                                     transpose_in0=False,
                                                     transpose_in1=True)
            graph.add(ct_1_wci_matmul_op, input_names=[Ct_1, Wci_name],
                      output_names=[ct_1_wci_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

            elementsum_of_term1_part2_op_name = lstm_node_name + "_elementsum_of_term1_part2_op"
            elementsum_of_term1_part2_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_part2_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(elementsum_of_term1_part2_op, input_names=[elementsum_of_term1_part1_bias_op_name, ct_1_wci_matmul_op_name],
                                output_names=[elementsum_of_term1_part2_op_name], idx=graph.nodes_in_order.index(lstm_node))

            inputs_to_activation_input_op = [elementsum_of_term1_part2_op_name]
        else:
            inputs_to_activation_input_op = [elementsum_of_term1_part1_bias_op_name]

        activation_input_op_name = lstm_node_name + "_activation_input_op"
        activation_input_op = op_adapter.NeuronOp(name=activation_input_op_name, neuron_type=ir_graph.QNN_OP_SIGMOID)
        graph.add(activation_input_op, input_names=inputs_to_activation_input_op,
                            output_names=[activation_input_op_name], idx=graph.nodes_in_order.index(lstm_node))

        return activation_input_op_name

    # ft = f(Xt*(Wxf^T) + Ht_1*(Whf^T) + ct_1*(Wcf^T) + Bf
    def expand_lstm_forget_gate(self, graph, lstm_node, Xt, Ht_1, Ct_1):
        lstm_node_name = lstm_node.op.name
        Wxf_idx, Whf_idx, Wcf_idx, Bf_idx = 1, 4, 19, 7

        Wxf_name = lstm_node.input_names[Wxf_idx]
        Whf_name = lstm_node.input_names[Whf_idx]
        Wcf_name = lstm_node.input_names[Wcf_idx]
        Bf_name = lstm_node.input_names[Bf_idx]
        Bf_size = graph.get_buffer(Bf_name).shape[-1]

        use_peephole_optimization = (Wcf_name != '')

        xt_wxf_matmul_op_name = lstm_node_name + "_xt_wxf_matmul_op"
        Wbx =  np.zeros(Bf_size, dtype=np.float32)
        xt_wxf_matmul_op = op_adapter.MatMulOp(name=xt_wxf_matmul_op_name,
                                                bias=Wbx,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(xt_wxf_matmul_op, input_names=[Xt, Wxf_name],
                  output_names=[xt_wxf_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        ht_1_whf_matmul_op_name = lstm_node_name + "_ht_1_whf_matmul_op"
        Wbh =  np.zeros(Bf_size, dtype=np.float32)
        ht_1_whf_matmul_op = op_adapter.MatMulOp(name=ht_1_whf_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(ht_1_whf_matmul_op, input_names=[Ht_1, Whf_name],
                  output_names=[ht_1_whf_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_of_term2_part1_op_name = lstm_node_name + "_elementsum_of_term2_part1_op"
        elementsum_of_term2_part1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_part1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term2_part1_op, input_names=[xt_wxf_matmul_op_name, ht_1_whf_matmul_op_name],
                            output_names=[elementsum_of_term2_part1_op_name], idx=graph.nodes_in_order.index(lstm_node))

        inputs_to_elementsum_of_term2_bias_op = [elementsum_of_term2_part1_op_name, Bf_name]

        if(use_peephole_optimization):
            ct_1_wcf_matmul_op_name = lstm_node_name + "_ct_1_wcf_matmul_op"
            Wbc =  np.zeros(Bf_size, dtype=np.float32)
            ct_1_wcf_matmul_op = op_adapter.MatMulOp(name=ct_1_wcf_matmul_op_name,
                                                     bias=Wbc,
                                                     transpose_in0=False,
                                                     transpose_in1=True)
            graph.add(ct_1_wcf_matmul_op, input_names=[Ct_1, Wcf_name],
                      output_names=[ct_1_wcf_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

            elementsum_of_term2_part2_op_name = lstm_node_name + "_elementsum_of_term2_part2_op"
            elementsum_of_term2_part2_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_part2_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(elementsum_of_term2_part2_op, input_names=[elementsum_of_term2_part1_op_name, ct_1_wcf_matmul_op_name],
                                output_names=[elementsum_of_term2_part2_op_name], idx=graph.nodes_in_order.index(lstm_node))

            inputs_to_elementsum_of_term2_bias_op = [elementsum_of_term2_part2_op_name, Bf_name]


        elementsum_of_term2_bias_op_name = lstm_node_name + "_elementsum_of_term2_bias_op"
        elementsum_of_term2_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term2_bias_op, input_names=inputs_to_elementsum_of_term2_bias_op,
                            output_names=[elementsum_of_term2_bias_op_name], idx=graph.nodes_in_order.index(lstm_node))

        activation_forget_op_name = lstm_node_name + "_activation_forget_op"
        activation_forget_op = op_adapter.NeuronOp(name=activation_forget_op_name, neuron_type=ir_graph.QNN_OP_SIGMOID)
        graph.add(activation_forget_op, input_names=[elementsum_of_term2_bias_op_name],
                            output_names=[activation_forget_op_name], idx=graph.nodes_in_order.index(lstm_node))

        return activation_forget_op_name

    # Ct = clip( ft(.)Ct_1 + it(.)g(Xt*(Wxc^T) + Ht_1*(Whc^T) + Bc) , tcell )
    def update_lstm_cell_gate(self, graph, lstm_node, Xt, Ht_1, Ct_1, it, ft):
        lstm_node_name = lstm_node.op.name

        Wxc_idx, Whc_idx, Bc_idx = 2, 5, 8
        Wxc_name = lstm_node.input_names[Wxc_idx]
        Whc_name = lstm_node.input_names[Whc_idx]
        Bc_name = lstm_node.input_names[Bc_idx]
        Bc_size = graph.get_buffer(Bc_name).shape[-1]

        ft_dot_ct_1_op_name = lstm_node_name + "_ft_dot_ct_1_op"
        ft_dot_ct_1_op = op_adapter.ElementwiseBinaryOp(name=ft_dot_ct_1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
        graph.add(ft_dot_ct_1_op, input_names=[ft, Ct_1],
                  output_names=[ft_dot_ct_1_op_name], idx=graph.nodes_in_order.index(lstm_node))

        xt_wxc_matmul_op_name = lstm_node_name + "_xt_wxc_matmul_op"
        Wbx =  np.zeros(Bc_size, dtype=np.float32)
        xt_wxc_matmul_op = op_adapter.MatMulOp(name=xt_wxc_matmul_op_name,
                                                bias=Wbx,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(xt_wxc_matmul_op, input_names=[Xt, Wxc_name],
                  output_names=[xt_wxc_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        ht_1_whc_matmul_op_name = lstm_node_name + "_ht_1_whc_matmul_op"
        Wbh =  np.zeros(Bc_size, dtype=np.float32)
        ht_1_whc_matmul_op = op_adapter.MatMulOp(name=ht_1_whc_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(ht_1_whc_matmul_op, input_names=[Ht_1, Whc_name],
                  output_names=[ht_1_whc_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_xt_wxc_matmul_ht_1_whc_matmul_op_name = lstm_node_name + "_elementsum_xt_wxc_matmul_ht_1_whc_matmul_op"
        elementsum_xt_wxc_matmul_ht_1_whc_matmul_op = op_adapter.ElementwiseBinaryOp(name=elementsum_xt_wxc_matmul_ht_1_whc_matmul_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_xt_wxc_matmul_ht_1_whc_matmul_op, input_names=[xt_wxc_matmul_op_name, ht_1_whc_matmul_op_name],
                            output_names=[elementsum_xt_wxc_matmul_ht_1_whc_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name = lstm_node_name + "_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op"
        elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op, input_names=[elementsum_xt_wxc_matmul_ht_1_whc_matmul_op_name, Bc_name],
                            output_names=[elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name], idx=graph.nodes_in_order.index(lstm_node)) 

        activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name = lstm_node_name + "_activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op"
        activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op = op_adapter.NeuronOp(name=activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name, neuron_type=ir_graph.QNN_OP_TANH)
        graph.add(activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op, input_names=[elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name],
                        output_names=[activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name], idx=graph.nodes_in_order.index(lstm_node))

        it_dot_activation_op_name = lstm_node_name + "_it_dot_activation_op"
        it_dot_activation_op = op_adapter.ElementwiseBinaryOp(name=it_dot_activation_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
        graph.add(it_dot_activation_op, input_names=[it, activation_elementsum_xt_wxc_matmul_ht_1_whc_matmul_bias_op_name],
                  output_names=[it_dot_activation_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_of_term3_op_name = lstm_node_name + "_elementsum_of_term3_op"
        elementsum_of_term3_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term3_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term3_op, input_names=[ft_dot_ct_1_op_name, it_dot_activation_op_name],
                            output_names=[elementsum_of_term3_op_name], idx=graph.nodes_in_order.index(lstm_node))

        tcell = np.float32(lstm_node.op.cell_clip_threshold)
        if not tcell:
            return elementsum_of_term3_op_name
        else :
            clip_elementsum_of_term3_op_name = lstm_node_name + "_clip_elementsum_of_term3_op"
            attrs = dict()
            attrs['min_value'] = -tcell
            attrs['max_value'] = tcell
            clip_elementsum_of_term3_op = op_adapter.NeuronOp(name=clip_elementsum_of_term3_op_name, neuron_type=ir_graph.QNN_OP_RELU_MIN_MAX, **attrs)
            graph.add(clip_elementsum_of_term3_op, input_names=[elementsum_of_term3_op_name],
                        output_names=[clip_elementsum_of_term3_op_name], idx=graph.nodes_in_order.index(lstm_node))

            return clip_elementsum_of_term3_op_name

    # ot = f(Xt*(Wxo^T) + Ht_1*(Who^T) + Ct*(Wco^T) + Bo)
    def expand_lstm_output_gate(self, graph, lstm_node, Xt, Ht_1, Ct):
        lstm_node_name = lstm_node.op.name

        Wxo_idx, Who_idx, Wco_idx, Bo_idx = 3, 6, 20, 9
        Wxo_name = lstm_node.input_names[Wxo_idx]
        Who_name = lstm_node.input_names[Who_idx]
        Wco_name = lstm_node.input_names[Wco_idx]
        Bo_name = lstm_node.input_names[Bo_idx]
        Bo_size = graph.get_buffer(Bo_name).shape[-1]

        use_peephole_optimization = (Wco_name != '')

        xt_wxo_matmul_op_name = lstm_node_name + "_xt_wxo_matmul_op"
        Wbx =  np.zeros(Bo_size, dtype=np.float32)
        xt_wxo_matmul_op = op_adapter.MatMulOp(name=xt_wxo_matmul_op_name,
                                                bias=Wbx,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(xt_wxo_matmul_op, input_names=[Xt, Wxo_name],
                  output_names=[xt_wxo_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        ht_1_who_matmul_op_name = lstm_node_name + "_ht_1_who_matmul_op"
        Wbh =  np.zeros(Bo_size, dtype=np.float32)
        ht_1_who_matmul_op = op_adapter.MatMulOp(name=ht_1_who_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        graph.add(ht_1_who_matmul_op, input_names=[Ht_1, Who_name],
                  output_names=[ht_1_who_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

        elementsum_of_term4_part1_op_name = lstm_node_name + "_elementsum_of_term4_part1_op"
        elementsum_of_term4_part1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term4_part1_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term4_part1_op, input_names=[xt_wxo_matmul_op_name, ht_1_who_matmul_op_name],
                            output_names=[elementsum_of_term4_part1_op_name], idx=graph.nodes_in_order.index(lstm_node))

        inputs_to_elementsum_of_term4_bias_op = [elementsum_of_term4_part1_op_name, Bo_name]

        if(use_peephole_optimization):
            ct_wco_matmul_op_name = lstm_node_name + "_ct_wco_matmul_op"
            Wbc =  np.zeros(Bo_size, dtype=np.float32)
            ct_wco_matmul_op = op_adapter.MatMulOp(name=ct_wco_matmul_op_name,
                                                     bias=Wbc,
                                                     transpose_in0=False,
                                                     transpose_in1=True)
            graph.add(ct_wco_matmul_op, input_names=[Ct, Wco_name],
                      output_names=[ct_wco_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

            elementsum_of_term4_part2_op_name = lstm_node_name + "_elementsum_of_term4_part2_op"
            elementsum_of_term4_part2_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term4_part2_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(elementsum_of_term4_part2_op, input_names=[elementsum_of_term4_part1_op_name, ct_wco_matmul_op_name],
                                output_names=[elementsum_of_term4_part2_op_name], idx=graph.nodes_in_order.index(lstm_node))

            inputs_to_elementsum_of_term4_bias_op = [elementsum_of_term4_part2_op_name, Bo_name]


        elementsum_of_term4_bias_op_name = lstm_node_name + "_elementsum_of_term4_bias_op"
        elementsum_of_term4_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term4_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
        graph.add(elementsum_of_term4_bias_op, input_names=inputs_to_elementsum_of_term4_bias_op,
                            output_names=[elementsum_of_term4_bias_op_name], idx=graph.nodes_in_order.index(lstm_node))

        activation_output_op_name = lstm_node_name + "_activation_output_op"
        activation_output_op = op_adapter.NeuronOp(name=activation_output_op_name, neuron_type=ir_graph.QNN_OP_SIGMOID)
        graph.add(activation_output_op, input_names=[elementsum_of_term4_bias_op_name],
                            output_names=[activation_output_op_name], idx=graph.nodes_in_order.index(lstm_node))

        return activation_output_op_name

    # Ht = clip( (ot(.)g(Ct) * Wproj^T) + Bproj , tproj ) # incase of projection ........(5.1)
    # Ht = ot(.)g(Ct) # otherwise ........(5.2)
    def update_lstm_hidden_state(self, graph, lstm_node, Ct, ot):
        lstm_node_name = lstm_node.op.name

        activation_ct_op_name = lstm_node_name + "_activation_ct_op"
        activation_ct_op = op_adapter.NeuronOp(name=activation_ct_op_name, neuron_type=ir_graph.QNN_OP_TANH)
        graph.add(activation_ct_op, input_names=[Ct],
                    output_names=[activation_ct_op_name], idx=graph.nodes_in_order.index(lstm_node))

        ot_dot_activation_ct_op_name = lstm_node_name + "_ot_dot_activation_ct_op"
        ot_dot_activation_ct_op = op_adapter.ElementwiseBinaryOp(name=ot_dot_activation_ct_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
        graph.add(ot_dot_activation_ct_op, input_names=[ot, activation_ct_op_name],
                    output_names=[ot_dot_activation_ct_op_name], idx=graph.nodes_in_order.index(lstm_node))

        Wproj_idx, Bproj_idx = 22, 23
        Wproj_name = lstm_node.input_names[Wproj_idx]
        Bproj_name = lstm_node.input_names[Bproj_idx]

        tproj = np.float32(lstm_node.op.output_clip_threshold)
        use_projections = (tproj != 0)

        if use_projections :
            Bproj_size = graph.get_buffer(Bproj_name).shape[-1]

            ot_dot_activation_ct_wproj_matmul_op_name = lstm_node_name + "_ot_dot_activation_ct_wproj_matmul_op_name"
            Wbproj = np.zeros(Bproj_size, dtype=np.float32)
            ot_dot_activation_ct_wproj_matmul_op = op_adapter.MatMulOp(name=ot_dot_activation_ct_wproj_matmul_op_name,
                                                                        bias=Wbproj,
                                                                        transpose_in0=False,
                                                                        transpose_in1=True)
            graph.add(ot_dot_activation_ct_wproj_matmul_op, input_names=[ot_dot_activation_ct_op_name, Wproj_name],
                        output_names=[ot_dot_activation_ct_wproj_matmul_op_name], idx=graph.nodes_in_order.index(lstm_node))

            ot_dot_activation_ct_wproj_matmul_bias_op_name = lstm_node_name + "_ot_dot_activation_ct_wproj_matmul_bias_op_name"
            ot_dot_activation_ct_wproj_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=ot_dot_activation_ct_wproj_matmul_bias_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            graph.add(ot_dot_activation_ct_wproj_matmul_bias_op, input_names=[ot_dot_activation_ct_wproj_matmul_op_name, Bproj_name],
                        output_names=[ot_dot_activation_ct_wproj_matmul_bias_op_name],  idx=graph.nodes_in_order.index(lstm_node))

            clip_proj_op_name = lstm_node_name + "_clip_proj_op"
            attrs = dict()
            attrs['min_value'] = -tproj
            attrs['max_value'] = tproj
            clip_proj_op = op_adapter.NeuronOp(name=clip_proj_op_name, neuron_type=ir_graph.QNN_OP_RELU_MIN_MAX, **attrs)
            graph.add(clip_proj_op, input_names=[ot_dot_activation_ct_wproj_matmul_bias_op_name],
                        output_names=[clip_proj_op_name], idx=graph.nodes_in_order.index(lstm_node))

            return clip_proj_op_name

        else :
            return ot_dot_activation_ct_op_name

    def expand_lstm_op_structure(self, graph):
        DATA_IDX, HIDDEN_IN_IDX, CELL_IN_IDX = 0, 10, 11

        sequence = [
            (op_adapter.LstmOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence)
        for nodes_tuple in matched_node_list:
            lstm_node = nodes_tuple[0]
            lstm_node_name = lstm_node.op.name
            lstm_node_idx = graph.nodes_in_order.index(lstm_node)

            batch_size, input_size = graph.get_buffer(lstm_node.input_names[DATA_IDX]).shape[:]
            output_size = graph.get_buffer(lstm_node.output_names[0]).shape[-1]
            num_units = graph.get_buffer(lstm_node.output_names[1]).shape[-1]

            Xt = lstm_node.input_names[DATA_IDX]
            Ht_1 = lstm_node.input_names[HIDDEN_IN_IDX]
            if len(graph.get_buffer(Ht_1).shape) != 2:
                Ht_1 = lstm_node_name + "_" + lstm_node.input_names[HIDDEN_IN_IDX] + "_reshape"
                input_h_reshape_output_shape = [batch_size, output_size]
                input_h_reshape_op = op_adapter.ReshapeOp(name=Ht_1,
                                                          shape=input_h_reshape_output_shape)
                graph.add(input_h_reshape_op, input_names=[lstm_node.input_names[HIDDEN_IN_IDX]],
                             output_names=[Ht_1], idx=graph.nodes_in_order.index(lstm_node))

            Ct_1 = lstm_node.input_names[CELL_IN_IDX]
            if len(graph.get_buffer(Ct_1).shape) != 2:
                Ct_1 = lstm_node_name + "_" + lstm_node.input_names[CELL_IN_IDX] + "_reshape"
                input_c_reshape_output_shape = [batch_size, num_units]
                input_c_reshape_op = op_adapter.ReshapeOp(name=Ct_1,
                                                          shape=input_c_reshape_output_shape)
                graph.add(input_c_reshape_op, input_names=[lstm_node.input_names[CELL_IN_IDX]],
                             output_names=[Ct_1], idx=graph.nodes_in_order.index(lstm_node))

            # expand lstm op structure
            # it = f(Xt*(Wxi^T) + Ht_1*(Whi^T) + Ct_1*(Wci^T) + Bi) ........(1) CIFG and peephole optimizations possible
            # ft = f(Xt*(Wxf^T) + Ht_1*(Whf^T) + Ct_1*(Wcf^T) + Bf ........(2) peephole optimizations possible
            # Ct = clip( ft(.)Ct_1 + it(.)g(Xt*(Wxc^T) + Ht_1*(Whc^T) + Bc) , tcell ) ........(3)
            # ot = f(Xt*(Wxo^T) + Ht_1*(Who^T) + Ct*(Wco^T) + Bo) ........(4) peephole optimizations possible
            # Ht = clip( (ot(.)g(Ct) * Wproj^T) + Bproj , tproj ) # incase of projection ........(5.1)
            # Ht = ot(.)g(Ct) # otherwise ........(5.2)

            ft = self.expand_lstm_forget_gate(graph, lstm_node, Xt, Ht_1, Ct_1) # ........(2)
            it = self.expand_lstm_input_gate(graph, lstm_node, Xt, Ht_1, Ct_1, ft) # ........(1)
            Ct = self.update_lstm_cell_gate(graph, lstm_node, Xt, Ht_1, Ct_1, it, ft) # ........(3)
            ot = self.expand_lstm_output_gate(graph, lstm_node, Xt, Ht_1, Ct) # ........(4)
            Ht = self.update_lstm_hidden_state(graph, lstm_node, Ct, ot) # ........(5)

            ht_out_idx, ct_out_idx, ot_out_idx = 0, 1, 2

            ht_buffer = graph.get_buffer(Ht)
            for consumer in list(graph.get_buffer(lstm_node.output_names[ht_out_idx]).consumers):
                ht_buffer.consumers.add(consumer)
                # insert the new buffer at the correct index in the consumer so that the order of nodes is not messed up.
                consumer.input_names.insert(consumer.input_names.index(lstm_node.output_names[ht_out_idx]), Ht)

            ct_buffer = graph.get_buffer(Ct)
            for consumer in list(graph.get_buffer(lstm_node.output_names[ct_out_idx]).consumers):
                ct_buffer.consumers.add(consumer)
                consumer.input_names.insert(consumer.input_names.index(lstm_node.output_names[ct_out_idx]), Ct)

            ot_buffer = graph.get_buffer(ot)
            for consumer in list(graph.get_buffer(lstm_node.output_names[ot_out_idx]).consumers):
                ot_buffer.consumers.add(consumer)
                consumer.input_names.insert(consumer.input_names.index(lstm_node.output_names[ot_out_idx]), ot)

            # prune original lstm_node
            graph.prune(lstm_node, force_remove=True)

@register_layer_optimization
class OptimizeScatterElementsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScatterElementsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name, updates_name = node.input_names
        input_buf,  indices_buf,  updates_buf = graph.get_input_buffers(node)

        output_buf = graph.get_output_buffers(node)[0]

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                return
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        # Check if any of the buffers has been changed into NDHWC, NSC, NFC, NTF order and revert if so
        # All inputs need to be in source framework order
        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])
        set_input_axis_format(updates_name, updates_buf.axis_format, node.op.data_axis_formats[2])

        input_buf = graph.get_input_buffers(node)[0]
        # set output buf axis format to input[0] axis format since data buffer's one is unchanged with ScatterElements
        output_buf.axis_format = input_buf.axis_format

        return True


@register_layer_optimization
class OptimizeScatterNDTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScatterNDOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name, updates_name = node.input_names
        input_buf,  indices_buf,  updates_buf = graph.get_input_buffers(node)

        output_buf = graph.get_output_buffers(node)[0]

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                return
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        # Check if any of the buffers have been changed into NDHWC, NSC, NFC, NTF order and revert if so
        # All inputs need to be in source framework order
        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])
        set_input_axis_format(updates_name, updates_buf.axis_format, node.op.data_axis_formats[2])

        input_buf = graph.get_input_buffers(node)[0]
        # set output buf axis format to input[0] axis format since data format is unchanged with ScatterND
        output_buf.axis_format = input_buf.axis_format

        return True


@register_layer_optimization
class OptimizeSplitTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SplitOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            axis_map = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        return True

    @staticmethod
    def remove_identity(node, graph):
        if not len(node.op.split_index):
            graph.squash(node, input_name=node.input_names[0])


@register_layer_optimization
class OptimizeSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # NB will probably want to switch to 'eltwise' version when we
        # support axis parameter.
        input_buf = graph.get_buffer(node.input_names[0])
        if AxisTracker.input_axis_formats_intact(graph, node) and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC]:
            # No change
            return False

        # Added this check for any 4D input for frcnn_vgg_compressed model
        # where it expects a permute after reshape
        if (input_buf.rank() == 5 and input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL and node.op.axis == 4) or \
            (input_buf.rank() == 4 and input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL and node.op.axis == 3):
            log_debug("Unsupported axis param {} in native axis format, don't permute".format(node.op.axis))
            output_buf = graph.get_buffer(node.output_names[0])
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.rank() == 4 and input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            AxisTracker.image_to_channel_last_order(node, graph)
        else:
            AxisTracker.alter_axis_format_to_ir_order(node, graph)

        # Ensure we're using the correct input buffer as a permute might have been inserted above
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            axis_map = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            log_debug('Mapping axis from {} to {}: '.format(node.op.axis, axis_map[node.op.axis]))
            node.op.axis = axis_map[node.op.axis]
        return True


@register_layer_optimization
class OptimizeUdlTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UdlOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_names = node.input_names
        for input_name in input_names:
            input_buf = graph.get_buffer(input_name)
            current_input_order = input_buf.get_axis_annotations()
            expected_input_order = []
            for dims in node.op.expected_input_axis_orders:
                if len(dims) == input_buf.rank():
                    expected_input_order = dims
            target_input_type = AxisTracker.get_axis_format_from_annotation(expected_input_order)
            permute_order = AxisTracker.compute_permute_order(current_input_order, expected_input_order)
            if len(permute_order) and permute_order != list(range(len(permute_order))):
                graph.inject_implicit_permute(input_name, target_input_type,
                                              permute_order, [node.op.name])

            target_output_order = []
            output_buffers = graph.get_output_buffers(node)
            for output_buf in output_buffers:
                for dims in node.op.expected_output_axis_orders:
                    if len(dims) == output_buf.rank():
                        target_output_order = dims
                output_buf.axis_format = AxisTracker.get_axis_format_from_annotation(target_output_order)
        return True


@register_layer_optimization
class OptimizeCropAndResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropAndResizeOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExtractGlimpseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExtractGlimpseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExtractPatchesTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExtractPatchesOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeImageProjectiveTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ImageProjectiveTransformOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMomentTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MomentOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMultiClassNmsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MultiClassNmsOp.TRANSLATION_KEY
        self.register_method(ADJUST_NMS_FEATURE_DIMS, self.adjust_nms_feature_dimensions)

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        for i, input_buf in enumerate(input_bufs):
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        input_bufs = graph.get_input_buffers(node)
        boxes_batch, num_boxes = input_bufs[0].shape[:2]

        for i, input_buf in enumerate(input_bufs):
            # the first input is boxes, the other inputs need to be 3D but NONTRIVIAL
            if (i == 0):
                continue
            # handle case where buf in NON TRIVIAL but not in expected order
            if input_buf.rank() == 3 and input_buf.shape[2] == num_boxes:
                graph.inject_implicit_permute(node.input_names[i], AxisTracker.AxisFormat.NONTRIVIAL,
                                              [0, 2, 1], [node.op.name])
                input_buf = graph.get_input_buffers(node)[i]
            # verify each input meets spec [batch, num_boxes] spec
            log_assert(input_buf.shape[:2] == [boxes_batch, num_boxes],
                       "Unable to get proper axis order for {} to expected prefix [batch, num_boxes]. Cannot match "
                       "input shapes [{}] with boxes input shapes [{}] for nms node {}."
                       .format(input_buf.name, input_buf.shape, input_bufs[0].shape, node.op.name))

        return True

    @staticmethod
    def adjust_nms_feature_dimensions(graph):
        """
        By default nms requires 2 inputs for boxes and score whose input and output shape is handled in
        TF translation. With the extra input_features they do not typically come with batch dimensions, so handle
        here by verifying required second dimension equality with num_boxes
        TODO: remove once backend consolidate input/output shapes of features to MultiClassNms. This should be
        handled during TF translation similar to the boxes and scores input.
        """

        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            # adjustment of features only needed if features are given as inputs
            if len(nms_node_.input_names) > 2 and len(nms_node_.output_names) > 4 and \
                    "scale_y" not in nms_node_.op.attrs:
                return True
            return False

        sequence = [
            (ir_graph.QNN_OP_MULTI_CLASS_NMS,
             (),
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_input_names = nms_node.input_names
            nms_output_names = nms_node.output_names
            num_boxes = graph.get_buffer(nms_node.input_names[0]).shape[1]
            for i in range(2, len(nms_node.input_names)):
                input_feature_buf = graph.get_buffer(nms_input_names[i])
                input_feature_shape = input_feature_buf.shape
                if len(input_feature_shape) == 1 or input_feature_shape[1] != num_boxes:
                    input_feature_node = graph.get_producer_node(nms_input_names[i])
                    # add reshape node to add batch dimension to the input features
                    expected_input_feature_shape = [1, *input_feature_shape]
                    # verify this is will result in expected input
                    log_assert(expected_input_feature_shape[1] == num_boxes,
                               "Unable to adjust input feature to match expected num_boxes on second dimension. "
                               "Got: {}, Expected num_boxes {}".format(expected_input_feature_shape, num_boxes))

                    if input_feature_node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY and \
                            graph.get_buffer(input_feature_node.input_names[0]).shape == expected_input_feature_shape:
                        # there was a squeeze done to remove batch dim, remove it and adjust to expected
                        # input feature instead.
                        graph.squash(input_feature_node, input_name=input_feature_node.input_names[0])
                        graph.get_buffer(input_feature_node.output_names[0]).set_buf_dims(expected_input_feature_shape)
                    else:
                        # add the reshape to add batch dim
                        input_feature_reshape_node_name = nms_input_names[i] + "_reshape_batch_add"
                        input_feature_reshape_op = op_adapter.ReshapeOp(name=input_feature_reshape_node_name,
                                                                        shape=expected_input_feature_shape)
                        graph.inject(input_feature_reshape_op, input_name=nms_input_names[i],
                                     output_name=input_feature_reshape_node_name,
                                     consumer_names=[nms_node.op.name])

                    # since we are reshaping input, output from nms will need to be adjusted as intermediate and
                    # will require a post reshape to remove batch dimension added.
                    output_name_idx = i + 2  # accounting for class and num_det output
                    output_feature_name = nms_output_names[output_name_idx]
                    output_feature_buf = graph.get_buffer(output_feature_name)
                    # replace the nms output as intermediate and the post reshaped output as the src fw output_feature
                    graph.delete_buffer(output_feature_name)
                    output_feature_reshape_op = op_adapter.ReshapeOp(name=output_feature_name,
                                                                     shape=output_feature_buf.shape)
                    # adjust to expected buffer shape for nms feature output(i.e with batch dim) and rename buffer as
                    # intermediate
                    output_feature_buf.set_buf_dims([1, *output_feature_buf.shape])
                    intermediate_output_name = output_feature_name + "_intermediate"
                    output_feature_buf.name = intermediate_output_name
                    graph.add_buffer(output_feature_buf)
                    nms_output_names[output_name_idx] = intermediate_output_name
                    graph.inject(output_feature_reshape_op, input_name=intermediate_output_name,
                                 output_name=output_feature_name)

                    # Addition of a const tensor to features should not be quantized
                    # TODO: add conditional that it should be set non quantizable based on tensortype and
                    #       quantization info of input tensor when irgraph supports these info
                    output_feature_reshape_buf = graph.get_buffer(output_feature_name)
                    for consumer in output_feature_reshape_buf.consumers:
                        if isinstance(consumer.op, op_adapter.ElementwiseBinaryOp):
                            for input_name in consumer.input_names:
                                eltwise_input_node = graph.get_producer_node(input_name)
                                if eltwise_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                                    eltwise_input_node.op.quantizable = False


@register_layer_optimization
class OptimizeNonMaxSuppressionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NonMaxSuppressionOp.TRANSLATION_KEY
        self.register_method(MERGE_LOW_LEVEL_OPS_TO_LAYERS, self.merge_low_level_ops_to_layers)

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        for i, input_buf in enumerate(input_bufs):
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        return True

    def merge_low_level_ops_to_layers(self, graph):
        validate_node = None

        sequence1 = [
            (ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            (ir_graph.QNN_OP_GATHER,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_NON_MAX_SUPPRESSION, "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_STRIDED_SLICE, "ALL")])
            ),
            (ir_graph.QNN_OP_STRIDED_SLICE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            )
        ]
        sequence2 = [
            (ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            (ir_graph.QNN_OP_GATHER,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_NON_MAX_SUPPRESSION, "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
        ]
        sequence3 = [
            (ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_STRIDED_SLICE, "ALL")])
            ),
            (ir_graph.QNN_OP_STRIDED_SLICE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_NON_MAX_SUPPRESSION, "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            )
        ]

        sequences = [sequence1, sequence2, sequence3]

        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node, ignore_constants=True)
            for node_tuple in matched_node_list:
                onnx_nms_node = node_tuple[0]
                if len(onnx_nms_node.output_names) != 1:
                    continue
                onnx_nms_op = onnx_nms_node.op
                nms_output_names = ['{}_boxes'.format(onnx_nms_op.name),
                                    '{}_scores'.format(onnx_nms_op.name),
                                    '{}_classes'.format(onnx_nms_op.name),
                                    '{}_num_detections'.format(onnx_nms_op.name)]

                nms_max_total_detections = onnx_nms_node.op.max_boxes_selected
                nms_iou_threshold = onnx_nms_node.op.iou_threshold
                nms_score_threshold = onnx_nms_node.op.score_threshold

                # Replace to MultiClassNmsOp
                nms_op_name = onnx_nms_op.name + '_gather'
                nms_op = op_adapter.MultiClassNmsOp(nms_op_name,
                                                    max_total_detections=nms_max_total_detections,
                                                    iou_threshold=nms_iou_threshold,
                                                    score_threshold=nms_score_threshold
                                                    )
                nms_input_names = onnx_nms_node.input_names.copy()
                last_node = node_tuple[-1]
                last_output_buf = graph.get_output_buffers(last_node)[0]

                pruned_nodes = []
                box_n_class_succors = []
                box_n_class_succors_input = []
                feature_consumer_succors = []
                for consumer in last_output_buf.consumers:
                    if consumer.op.type == ir_graph.QNN_OP_GATHER:
                        consumer_input_names = consumer.input_names
                        gather_data_inputs = [input_name for input_name in consumer_input_names if input_name != last_output_buf.name]
                        # boxes and classes nodes have been done in nms_output_names
                        # therefore no need to create an extra output from gather op
                        if gather_data_inputs[0] in nms_input_names[:2]:
                            box_n_class_succors_input.append(nms_output_names[nms_input_names.index(gather_data_inputs[0])])
                            box_n_class_succors.append(graph.get_output_buffers(consumer)[0].consumers)
                        # feature parts, which need to be added as extra outputs
                        # connected the graph by nms output[4:]
                        else:
                            nms_input_names.extend(gather_data_inputs)
                            nms_output_names.extend(consumer.output_names)
                            # gather has only one output buffer
                            feature_consumer_succors.append(graph.get_output_buffers(consumer)[0].consumers)
                        pruned_nodes.append(consumer)
                for node in pruned_nodes:
                    graph.prune(node, force_remove=True)

                # Prune the nodes after extract required information
                for node_in_tuple in reversed(node_tuple):
                    graph.prune(node_in_tuple, force_remove=True)
                idx_to_insert = 0
                for input_name in nms_input_names:
                    buf = graph.get_buffer(input_name)
                    cur_idx = graph.nodes_in_order.index(buf.producer)
                    if idx_to_insert <= cur_idx:
                        idx_to_insert = cur_idx + 1
                nms_node = graph.add(nms_op, input_names=nms_input_names, output_names=nms_output_names,idx=idx_to_insert)
                # re-connected the nodes after gather
                # box, scores part
                for idx, succs in enumerate(box_n_class_succors):
                    for succ_node in succs:
                        succ_node.input_names.append(box_n_class_succors_input[idx])
                        nms_output_buf = graph.get_buffer(nms_output_names[idx])
                        nms_output_buf.consumers.add(succ_node)
                # feature part
                for idx, succs in enumerate(feature_consumer_succors):
                    succ_input_name = nms_output_names[4+idx]
                    for succ_node in succs:
                        succ_node.input_names.append(succ_input_name)
                        nms_output_buf = graph.get_buffer(nms_output_names[4+idx])
                        nms_output_buf.consumers.add(succ_node)


@register_layer_optimization
class OptimizePackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PackOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        for idx, input_name in enumerate(node.input_names):
            input_buf = graph.get_buffer(input_name)
            # Pack needs to happen in src format, so in case the current input format and the data_axis_format
            # are different, inject permute to change it back to src format
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.NCDHW:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.NCS:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.NCF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.TNF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            else:
                log_debug2("No axes change for Op {}".format(node.op.name))


@register_layer_optimization
class OptimizeDepthToSpaceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DepthToSpaceOp.TRANSLATION_KEY
        self.register_method(MATCH_DEPTHTOSPACE, self.match_depthtospace)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.rank() != 4:
            raise ValueError("Backend only support DepthToSpace with rank 4, but got an input rank with {}."
                             .format(input_buf.rank()))
        # To ensure the depthToSpace's input and output format as NSC
        AxisTracker.image_to_channel_last_order(node, graph)
        return True

    @staticmethod
    def match_depthtospace(graph):
        # To validate the getting node tuple match the optimiaztion solution.
        def validate_node_tuple(node_tuple):
            for d2s_node in node_tuple[3:5]:
                if d2s_node.op.mode != ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR:
                    return False
            conv_op = node_tuple[0]
            conv_out_channel_idx = AxisTracker.get_axis_annotation_from_format(
                graph.get_buffer(conv_op.input_names[1]).axis_format
                ).index(AxisTracker.AxisAnnotations.OUTPUT_CHANNELS)
            split_node = node_tuple[2]
            split_axis = split_node.op.__getattr__(ir_graph.QNN_OP_SPLIT_PARAM_AXIS)
            concat_node = node_tuple[6]
            concat_axis = concat_node.op.__getattr__(ir_graph.QNN_OP_CONCAT_PARAM_AXIS)
            if split_axis != concat_axis or split_axis != conv_out_channel_idx:
                return False
            return True

        # rearrange the conv data to reorder the channel axis from CRD to DCR
        def rearrange_conv(weight, bias, conv_out_channel_idx, block_size):
            block_height = block_size[0]
            block_width = block_size[1]
            new_weight = weight.copy()
            new_bias = bias.copy()
            conv_out_channel_dim = weight.shape[conv_out_channel_idx]
            depth = int(conv_out_channel_dim / (block_height * block_width))
            idxes = np.zeros((block_height, block_width, depth), dtype=np.dtype("int32"))
            idx = 0
            # Reorder the channel axis from CRD to DCR
            for k in range(depth):
                for i in range(block_height):
                    for j in range(block_width):
                        idxes[i,j,k] = idx
                        idx = idx + 1
            idx_list = idxes.flatten()
            for i in range(conv_out_channel_dim):
                if weight.ndim == 4:
                    if conv_out_channel_idx == 1:
                        new_weight[:,i,:,:] = weight[:,idx_list[i],:,:]
                    if conv_out_channel_idx == 3:
                        new_weight[:,:,:,i] = weight[:,:,:,idx_list[i]]
                elif weight.ndim == 5:
                    if conv_out_channel_idx == 1:
                        new_weight[:,i,:,:,:] = weight[:,idx_list[i],:,:,:]
                    if conv_out_channel_idx == 4:
                        new_weight[:,:,:,:,i] = weight[:,:,:,:,idx_list[i]]
                new_bias[i] = bias[idx_list[i]]
            return new_weight, new_bias

        sequence = [
            (ir_graph.QNN_OP_CONV_2D,
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")])
             ),
            (ir_graph.QNN_OP_SPLIT,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY")])
             ),
            (ir_graph.QNN_OP_DEPTH_TO_SPACE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_CONCAT, "ALL")])
             ),
            (ir_graph.QNN_OP_DEPTH_TO_SPACE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_CONCAT, "ALL")])
             ),
            (ir_graph.QNN_OP_DEPTH_TO_SPACE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_CONCAT, "ALL")])
             ),
            (ir_graph.QNN_OP_CONCAT,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY")]),
             ()
             ),
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validate_node_tuple, ignore_constants=True)

        for node_tuple in matched_node_list:
            conv_op = node_tuple[0]
            conv_out_channel_idx = AxisTracker.get_axis_annotation_from_format(
                graph.get_buffer(conv_op.input_names[1]).axis_format
                ).index(AxisTracker.AxisAnnotations.OUTPUT_CHANNELS)
            add_op = node_tuple[1]
            split_node = node_tuple[2]
            block_size = node_tuple[3].op.block_size
            weight_producer = graph.get_producer_op(conv_op.input_names[1])
            bias_producer = graph.get_producer_op(conv_op.input_names[2])
            if bias_producer.tensor.max() == bias_producer.tensor.min() == 0:
                bias_producer = graph.get_producer_op(add_op.input_names[1])
            new_weights, new_bias = rearrange_conv(weight_producer.tensor, bias_producer.tensor, conv_out_channel_idx, block_size)
            weight_producer.tensor = new_weights
            bias_producer.tensor = new_bias

            # merge the three DCR D2S to one DCR D2s
            for node in node_tuple[:2:-1]:
                input_names = node.input_names[:]
                # pick squashable input based on whether current node is only consumer and input is not network input
                input_name = [name for name in input_names if (len(graph.get_buffer(name).consumers) == 1 and
                              not isinstance(graph.get_producer_op(name), op_adapter.InputOp))][0]
                input_names.remove(input_name)
                for input_name_ in input_names:
                    # disconnect rest of inputs from node
                    input_buf_ = graph.get_buffer(input_name_)
                    input_buf_.consumers.remove(node)
                    node.input_names.remove(input_name_)
                graph.squash(node, input_name=input_name)

            redundant_outputs = [name for name in split_node.output_names if len(graph.get_buffer(name).consumers) == 0]
            for name in redundant_outputs:
                graph.delete_buffer(name)
                split_node.output_names.remove(name)

            # replace the split + 3 DCR D2S op with 1 DCR D2S
            split_op = split_node.op
            split_op_name = graph.naming_policy.get_op_name(split_op)
            DCR_D2S_op_name = split_op_name + '_DCR_DepthToSpace'
            DCR_D2S_op = op_adapter.DepthToSpaceOp(DCR_D2S_op_name, block_size=block_size, mode=ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR)
            graph.replace(split_op, DCR_D2S_op)

    def merge_low_level_ops_to_layers(self, graph):
        # DCR mode: elements along the depth dimension are rearranged in the order of depth, column, and then row.
        #     input: [n, c, h, w]
        #     reshape: [n, blk_h, blk_w, c/(blk_h*blk_w), h, w]
        #     transpose: [n, c/(blk_h*blk_w), h, blk_h, w, blk_w] with [0, 3, 4, 1, 5, 2]
        #     reshape: [n, c/(blk_h*blk_w), h*blk_h, w*blk_w]
        #
        # CRD mode: elements along the depth dimension are rearranged in the order of column, row, and then depth.
        #     input: [n, c, h, w]
        #     reshape: [n, c/(blk_h*blk_w), blk_h, blk_w, h, w]
        #     transpose: [n, c/(blk_h*blk_w), h, blk_h, w, blk_w] with [0, 1, 4, 2, 5, 3]
        #     reshape: [n, c/(blk_h*blk_w), h*blk_h, w*blk_w]

        def validate(nodes_tuple_):
            reshape_to_6D_node = nodes_tuple_[0]
            permute_node = nodes_tuple_[1]
            reshape_to_4D_node = nodes_tuple_[2]

            reshape6d_input_shape = graph.get_input_shapes(reshape_to_6D_node)[0]
            reshape6d_output_shape = graph.get_output_shapes(reshape_to_6D_node)[0]
            reshape4d_input_shape = graph.get_input_shapes(reshape_to_4D_node)[0]
            reshape4d_output_shape = graph.get_output_shapes(reshape_to_4D_node)[0]

            # Check the output shape should be 4
            if len(reshape4d_output_shape) != 4:
                return False

            if len(reshape6d_output_shape) == 5:
                # Check the Channel dimension is split into blocks
                if np.prod(reshape6d_output_shape[1:3]) != reshape6d_input_shape[1]:
                    return False
                # Check the permute order for CRD or DCR mode
                if permute_node.op.perm not in [[0,1,3,4,2]]:
                    return False
                # Check that the block_size was reshaped into H and W
                if reshape4d_output_shape[2] != np.prod(reshape4d_input_shape[2:3]) or \
                        reshape4d_output_shape[3] != np.prod(reshape4d_input_shape[3:]):
                    return False
                return True

            if len(reshape6d_output_shape) == 6:
                # Check the Channel dimension is split into blocks
                if np.prod(reshape6d_output_shape[1:4]) != reshape6d_input_shape[1]:
                    return False
                # Check the permute order for CRD or DCR mode
                if permute_node.op.perm not in [[0,1,4,2,5,3], [0,3,4,1,5,2]]:
                    return False
                # Check that the block_size was reshaped into H and W
                if reshape4d_output_shape[2] != np.prod(reshape4d_input_shape[2:4]) or \
                        reshape4d_output_shape[3] != np.prod(reshape4d_input_shape[4:]):
                    return False
                return True

            return False

        sequence = [
                    (ir_graph.QNN_OP_RESHAPE, (), ()),
                    (ir_graph.QNN_OP_TRANSPOSE,
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_RESHAPE, 0)]),
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_RESHAPE, 0)]),),
                    (ir_graph.QNN_OP_RESHAPE,
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_TRANSPOSE, 0)]),
                        ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            reshape_to_6D_node = nodes_tuple[0]
            permute_node = nodes_tuple[1]
            reshape_to_4D_node = nodes_tuple[2]

            reshape4d_input_shape = graph.get_input_shapes(reshape_to_4D_node)[0]
            upscale_factor_height = 0
            upscale_factor_width = 0

            if len(reshape4d_input_shape) == 5:
                upscale_factor_height = 1
                upscale_factor_width = reshape4d_input_shape[4]
            elif len(reshape4d_input_shape) == 6:
                upscale_factor_height = reshape4d_input_shape[3]
                upscale_factor_width = reshape4d_input_shape[5]
            else:
                continue

            if permute_node.op.perm == [0, 1, 4, 2, 5, 3]:
                d2s_mode = ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD
            elif permute_node.op.perm == [0, 1, 3, 4, 2]:
                d2s_mode = ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD
            else:
                d2s_mode = ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR

            # Squashes the reshape_4d node
            reshape4d_input_buffer = graph.get_input_buffers(reshape_to_4D_node)[0]
            graph.squash(reshape_to_4D_node, input_name=reshape4d_input_buffer.name)

            # Squashes the permute node
            permute_input_buffer = graph.get_input_buffers(permute_node)[0]
            graph.squash(permute_node, input_name=permute_input_buffer.name)

            # Replace the reshape6D OpNode to a DepthToSpace OpNode
            d2s_op = op_adapter.DepthToSpaceOp(name=permute_node.op.name,
                                               block_size=[upscale_factor_height, upscale_factor_width],
                                               mode=d2s_mode)
            graph.replace(reshape_to_6D_node.op, d2s_op)


@register_layer_optimization
class OptimizeStridedSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StridedSliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeStridedSliceTranslation, self).axes_to_spatial_first_order(node, graph):
            # No change in input formats, and none of the input formats are NonTrivial
            return False

        input_buf = graph.get_buffer(node.input_names[0])

        spatial_last_axis_formats = [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF, AxisTracker.AxisFormat.TNF]
        spatial_first_axis_formats = [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC, AxisTracker.AxisFormat.NTF]

        # if data_axis_formats is spatial-last order and input_buf.axis_format is spatial-first, transform the attributes
        if (node.op.data_axis_formats[0], input_buf.axis_format) in list(zip(spatial_last_axis_formats, spatial_first_axis_formats)):

            begins, ends, strides = list(map(list, zip(*node.op.ranges.tolist())))

            # tranform begins/ends/strides from spatial-last format to spatial-first format
            begins = SpatialLastAxisOrder().permute_shape_to_ir(begins, input_buf.axis_format)
            ends = SpatialLastAxisOrder().permute_shape_to_ir(ends, input_buf.axis_format)
            strides = SpatialLastAxisOrder().permute_shape_to_ir(strides, input_buf.axis_format)

            ranges_data = np.array(list(map(list, zip(begins, ends, strides))), dtype=np.int32)
            ranges = ir_graph.IrStaticTensor(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                             list(ranges_data.shape),
                                             ranges_data,
                                             ir_graph.QNN_DATATYPE_INT_32)
            node.op.ranges = ranges

        return True


@register_layer_optimization
class OptimizeSpaceToDepthTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SpaceToDepthOp.TRANSLATION_KEY
        self.register_method(MATCH_SPACETODEPTH, self.match_spacetodepth)

    @staticmethod
    def match_spacetodepth(graph):
        # check shapes in the reshape layers
        # [n, c, h, w] -> [n, c * blk**2, h/blk, w/blk]
        def is_valid_spacetodepth(node_tuple):
            input_buf = graph.get_input_buffers(node_tuple[0])[0]
            input_shape = input_buf.shape
            first_reshape_output_shape = graph.get_output_shapes(node_tuple[0])[0]
            if len(input_shape) == 4 and len(first_reshape_output_shape) == 6:
                blocksize = first_reshape_output_shape[3]
                sequence_output_shape = graph.get_output_shapes(node_tuple[-1])[0]

                batch, height, width, channel = graph.src_axis_order.extract_2d_spatial_dims(input_shape)
                expected_shape = graph.src_axis_order.format_2d_spatial_output_shape(batch_size=batch,
                                                                                     channel=channel * (blocksize**2),
                                                                                     height=height//blocksize,
                                                                                     width=width//blocksize)
                return sequence_output_shape == expected_shape
            else:
                return False

        # reshape:   [n, c, h/blk1, blk1, w/blk2, blk2], blk1 == blk2, number is for transpose order.
        # transpose: [n, c, h/blk1, w/blk2, blk1, blk2]
        # reshape:   [n, c, h/blk * w/blk, blk ** 2]
        # transpose: [n, c, blk ** 2, h/blk * w/blk]
        # reshape:   [n, c, blk ** 2, h/blk, w/blk]
        # transpose: [n, blk ** 2, c, h/blk, w/blk]
        # reshape:   [n, c*(blk**2), h/blk, w/blk]
        sequence = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
            ),
            ("Transpose",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
            ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
            ),
            ("Transpose",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
            ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
            ),
            ("Transpose",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
            ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
             ()
            )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_spacetodepth, ignore_constants=True)
        for node_tuple in matched_node_list:
            downscale_factor = graph.get_output_shapes(node_tuple[0])[0][3]
            block_size = [downscale_factor] * 2
            reshape_node = node_tuple[0]
            # Squash all nodes except the first reshape in reverse order
            # the first reshape op will be replaced
            for node in node_tuple[:0:-1]:
                for input_name in node.input_names:
                    graph.squash(node, input_name=input_name)
            reshape_op = reshape_node.op
            reshape_op_name = reshape_op.name
            spacetodepth_op_name = reshape_op_name + '_space_to_depth'
            spacetodepth_op = op_adapter.SpaceToDepthOp(spacetodepth_op_name, block_size=block_size)
            graph.replace(reshape_op, spacetodepth_op)


@register_layer_optimization
class OptimizeSsdTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BoxDecoderOp.TRANSLATION_KEY
        self.register_method(SQUASH_BOX_DECODER, self.squash_box_decoder)

    @staticmethod
    def squash_box_decoder(graph):
        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            nms_input_names_ = nms_node_.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_input_names_[0]).type:
                # remove optional reshape input to check if previous is box decoder(ssd) below
                reshape_node_ = graph.get_producer_node(nms_node_.input_names[0])
                nms_input_names_ = [nms_input_names_[1], *reshape_node_.input_names]

            if any(op_adapter.BoxDecoderOp.TRANSLATION_KEY == graph.get_producer_node(name_).op.TRANSLATION_KEY
                   for name_ in nms_input_names_):
                return True

            return False

        sequence = [
                    (ir_graph.QNN_OP_MULTI_CLASS_NMS,
                        (),
                        ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_op = nms_node.op
            # update the boxes input of nms to be box decoder's inputs along with box decoder's op attributes.
            #  [boxes]_______[anchor or priorboxes]
            #            |
            #       [box_decoder(ssd_op)]   <- remove
            #                  |
            #        remove->([Reshape] (optional))_______[scores]
            #                                         |
            #                                 [non_max_suppression]   <- replace by [detection_output]
            #                                         |
            #                                   [detection_output]
            # Updated input for nms will be: [scores, boxes, anchor(priorboxes)]

            nms_boxes_input_name, nms_scores_input_name = nms_node.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_boxes_input_name).type:
                # update inputs for nms and subsequently the boxes_node
                reshape_node = graph.get_producer_node(nms_boxes_input_name)
                reshape_buf = graph.get_buffer(nms_boxes_input_name)
                nms_boxes_input_name = reshape_node.input_names[0]

                # update consumer relation with reshape buf and prune if applicable
                reshape_buf.consumers.remove(nms_node)
                if len(reshape_buf.consumers) == 0:
                    graph.prune(reshape_node)

            # fold box_decoder(ssd) node
            box_decoder_node = graph.get_producer_node(nms_boxes_input_name)
            box_decoder_buf = graph.get_buffer(nms_boxes_input_name)
            # Copy over input_names and all op attrs to nms op
            nms_node.input_names = [nms_scores_input_name, *box_decoder_node.input_names]

            # update consumer relation with nms node, box_decoder node and input to box_decoder and
            # prune if applicable
            for name in box_decoder_node.input_names:
                buf = graph.get_buffer(name)
                buf.consumers.add(nms_node)
            if nms_node in box_decoder_buf.consumers:
                box_decoder_buf.consumers.remove(nms_node)
            if len(box_decoder_buf.consumers) == 0:
                graph.prune(box_decoder_node)

            # replace nms_node(non_maximum_suppress)
            attrs = dict()
            attrs['delta_scaling_factors'] = [box_decoder_node.op.scale_y, box_decoder_node.op.scale_x, box_decoder_node.op.scale_h, box_decoder_node.op.scale_w]
            attrs['confidence_threshold'] = nms_op.score_threshold
            attrs['iou_threshold'] = nms_op.iou_threshold
            attrs['detection_limit'] = nms_op.max_total_detections
            attrs['use_bg_in_nms'] = 1
            output_dims = []
            for output_name in nms_node.output_names:
                output_dims.append(graph.get_buffer(output_name).shape)
            # nms outputs are [box, score, pred_cls, num_detection]
            # detection outputs [score, box, pred_cls, num_detection]
            output_dims = [
                output_dims[1],
                output_dims[0],
                output_dims[2],
                output_dims[3]
            ]
            attrs['output_dims'] = output_dims
            detection_op = op_adapter.DetectionOutputOp(name=nms_op.name, **attrs)
            graph.replace(nms_op, detection_op)
            detection_node = graph.get_node_by_name(detection_op.name)

            # nms outputs are [box, score, pred_cls, num_detection]
            # detection outputs [score, box, pred_cls, num_detection]
            detection_node.output_names = [
                detection_node.output_names[1],
                detection_node.output_names[0],
                detection_node.output_names[2],
                detection_node.output_names[3]
            ]

            # Update Anchors inputs to fit DetectionOut spec
            anchor_buf = graph.get_buffer(nms_node.input_names[-1])
            anchor_data = anchor_buf.producer.op.tensor

            # TF style (decodeBox+nms) comes as CORNER_SIZE spec requires CENTER_SIZE
            for batch in range(0, anchor_buf.shape[0]):
                for i in range(0, anchor_buf.shape[1]):
                    y_min, x_min, y_max, x_max = anchor_data[batch][i]
                    height = (y_max - y_min)
                    width = (x_max - x_min)
                    anchor_data[batch][i][0] = y_min + height / 2.  # center_y
                    anchor_data[batch][i][1] = x_min + width / 2.  # center_x
                    anchor_data[batch][i][2] = height  # height
                    anchor_data[batch][i][3] = width

            # Addition of a const tensor to class labels should not be quantized
            classes_buf = graph.get_buffer(nms_node.output_names[2])
            for consumer in classes_buf.consumers:
                if consumer.op.type == op_adapter.ElementwiseBinaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_ADD]:
                    for input_name in consumer.input_names:
                        add_input_node = graph.get_producer_node(input_name)
                        if add_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                            add_input_node.op.quantizable = False

            # change shape for anchor input from [batch, num_anchors, 4] to [batch * num_anchors, 4] per spec
            anchor_buf.shape = [anchor_buf.shape[0] * anchor_buf.shape[1], anchor_buf.shape[2]]
            anchor_buf.producer.op.tensor = anchor_data.reshape(anchor_buf.shape)

            log_debug2(code_to_message.get_debugging_message("DEBUG_BOXDECODER_SQUASH")(box_decoder_node.op.name,
                                                                                        nms_node.op.name))


@register_layer_optimization
class OptimizeTileTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TileOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            node.op.multiples = graph.src_axis_order.permute_shape_to_ir(node.op.multiples,
                                                                         input_buf.axis_format)

        return True


@register_layer_optimization
class OptimizeTopKTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TopKOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeTopKTranslation, self).axes_to_spatial_first_order(node, graph):
            return False

        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            axis_map = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]

        return True


@register_layer_optimization
class OptimizeUnpackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UnpackOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # Pack needs to happen in src format, so in case the current input format and the data_axis_format
        # are different, inject permute to change it back to src format
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                          AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                          AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                          AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                          AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
        else:
            log_debug2("No axes change for Unpack Op named {}".format(node.op.name))
