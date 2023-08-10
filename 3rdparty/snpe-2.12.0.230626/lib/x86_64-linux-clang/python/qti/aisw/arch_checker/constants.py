# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

INDEX_CHANNEL = 3

O_C_GRAPH_NODENAME = "Graph/Layer_name"
O_C_TYPE = "Type"
O_C_INPUTS = "Input_tensor_name:[dims]"
O_C_OUTPUTS = "Output_tensor_name:[dims]"
O_C_ISSUE = "Issue"
O_C_RECOMM = "Recommendation"
O_C_PARAM = "Parameters"
O_C_PRODUCER = "Previous layer"
O_C_CONSUMERS = "Next layers"
OUTPUT_CSV_HEADER=[O_C_GRAPH_NODENAME, O_C_ISSUE, O_C_RECOMM, O_C_TYPE, O_C_INPUTS, O_C_OUTPUTS, O_C_PARAM, O_C_PRODUCER, O_C_CONSUMERS]
