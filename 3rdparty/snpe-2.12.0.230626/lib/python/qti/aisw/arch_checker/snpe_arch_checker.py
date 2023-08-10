# =============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.arch_checker.arch_checker import ArchChecker

class SnpeArchChecker(ArchChecker):

    def __init__(self, c_ir_graph, constraints_json, out_path, logger, model_info):
        super(SnpeArchChecker, self).__init__(c_ir_graph, constraints_json, out_path, logger)
        self.model_info = model_info

    def is_8bit(self):
        if self.model_info.model_reader.quantizer_command == "N/A":
            return False
        if "act_bitwidth=[16]" in self.model_info.model_reader.quantizer_command:
            return False
        return True
