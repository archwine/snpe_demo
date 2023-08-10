# =============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import AxisFormat

def permute_tensor_data_axis_order(src_axis_format, axis_format, tensor_dims, golden_tensor_data):
    """
    Permutes intermediate tensors goldens to spatial-first axis order for verification
    :param src_axis_format: axis format of source framework tensor
    :param axis_format: axis format of QNN tensor
    :param tensor_dims: current dimensions of QNN tensor
    :param golden_tensor_data: golden tensor data to be permuted
    :return: np.array of permuted golden tensor data
    """

    # base case for same axis format / nontrivial
    if src_axis_format == axis_format or src_axis_format == 'NONTRIVIAL' or axis_format == 'NONTRIVIAL':
        return golden_tensor_data
    # reshape golden data to spatial-last axis format
    golden_tensor_data = np.reshape(golden_tensor_data, tuple([tensor_dims[i] for i in AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][0]]))
    # transpose golden data to spatial-first axis format
    golden_tensor_data = np.transpose(golden_tensor_data, AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][1])
    # return flatten golden data
    return golden_tensor_data.flatten()
