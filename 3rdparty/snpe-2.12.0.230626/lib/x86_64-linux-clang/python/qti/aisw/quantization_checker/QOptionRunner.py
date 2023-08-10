#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
from qti.aisw.quantization_checker.utils import utils
import qti.aisw.quantization_checker.utils.target as target
import qti.aisw.quantization_checker.EnvironmentManager as em
from qti.aisw.quantization_checker.utils.ConfigParser import extractEnvironmentConfigParams
from qti.aisw.quantization_checker.utils.Progress import Progress, ProgressStage

class QOptionRunner:
    def __init__(self, graphDir, inputList, sdkDir, outputDir, configFile, logger):
        self.quantizationOption = 'unquantized'
        self.logger = logger
        self.graphDir = graphDir
        self.inputList = inputList
        self.sdkDir = sdkDir
        self.outputDir = outputDir
        self.configParams = extractEnvironmentConfigParams(os.path.abspath(configFile))

    def run(self):
        if not os.path.isdir(self.graphDir):
            self.logger.print('Please enter a valid directory containing the model file and output from the QOptionGenerator.py script. Graph directory: ' + self.graphDir)
            return -1

        runResult = -1
        graphBasePathName = os.path.join(self.graphDir, self.quantizationOption)
        if not os.path.isfile(os.path.join(graphBasePathName, self.quantizationOption + '.dlc')):
            self.logger.print('The generated output files cannot be found in the model directory! If the generator output files were stored in a different location, please specify the location using the --output_dir option')
            return -1
        runResult = self.runModel()

        if runResult == -1:
            self.logger.print('Error encountered during running of ' + self.quantizationOption + ' quantization option. Please consult console/log output.')
            return runResult
        self.logger.print('Build and execute complete for ' + self.quantizationOption + '. Please refer to output files for accuracy comparision.\n')

        return runResult

    def runModel(self):
        originalDir = os.getcwd()
        utils.changeDir(self.graphDir)
        try:
            self.logger.print('outputDir: ' + self.outputDir)
            x86_64 = target.x86_64(self.quantizationOption, self.inputList, self.graphDir, self.sdkDir, self.outputDir, self.configParams)
            x86_64.buildSnpeNetRunArgs()
            result = x86_64.runModel(logger=self.logger)
            if result != 0:
                self.logger.print('snpe-net-run failed for unquantized. Please check the console or logs for details.')
                return -1
            #TODO: add different target to raise exceptions when there are errors while runModel so it can be caught in except. Otherwise we will always report back complete.
            self.logger.print('snpe-net-run complete, output saved in ' + self.outputDir + '\n')
        except Exception as e:
            self.logger.print("ERROR! running model failed: " + str(e))
            return -1
        utils.changeDir(originalDir)
        return 0
