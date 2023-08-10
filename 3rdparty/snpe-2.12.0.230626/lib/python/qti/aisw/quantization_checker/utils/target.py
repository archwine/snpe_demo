#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
from qti.aisw.quantization_checker.utils.Logger import Logger
from qti.aisw.quantization_checker.utils import utils
import qti.aisw.quantization_checker.EnvironmentManager as em
import qti.aisw.quantization_checker.utils.Constants as Constants

# contains general information about the target execution environment
class target:
    def __init__(self, quantizationVariation, inputList, graphDir, sdkDir, targetBasePath, outputDir, configParams):
        self.snpeNetRunCmdAndArgs = ''
        self.quantizationVariation = quantizationVariation
        self.inputList = inputList
        self.graphDir = graphDir
        self.sdkDir = sdkDir
        self.targetBasePath = targetBasePath
        self.outputDir = outputDir
        self.configParams = configParams
        self.unquantizedDlc = quantizationVariation + '.dlc'

# executes locally on x86 arch
class x86_64(target):
    def __init__(self, quantizationVariation, inputList, graphDir, sdkDir, outputDir, configParams):
        super().__init__(quantizationVariation, inputList, graphDir, sdkDir, Constants.BIN_PATH_IN_SDK, outputDir, configParams)

    def buildSnpeNetRunArgs(self):
        self.snpeNetRunCmdAndArgs = os.path.join(self.sdkDir, self.targetBasePath, Constants.NET_RUN_BIN_NAME) + ' --container ' + os.path.join(self.graphDir, self.quantizationVariation, self.unquantizedDlc) + ' --output_dir ' + os.path.join(self.outputDir, self.quantizationVariation) + ' --input_list ' + self.inputList + ' --debug'

    def runModel(self, logger: Logger):
        environment = em.getEnvironment(self.configParams, self.sdkDir)
        return utils.issueCommandAndWait(self.snpeNetRunCmdAndArgs, logger, environment, False)
