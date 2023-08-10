#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import sys
from qti.aisw.quantization_checker.utils.ConfigParser import extractEnvironmentConfigParams
import qti.aisw.quantization_checker.EnvironmentManager as em

class QOptionGenerator:
    def __init__(self, quantizationVariations, modelFile, inputList, sdkDir, activationWidth, biasWidth, weightWidth, inputDimension, outputDir, quantOverridesPath, config_file, logger):
        self.quantizationVariations = quantizationVariations
        self.quantAlgorithms = []
        self.quantOptions = []
        self.modelFile = modelFile
        self.inputList = inputList
        self.sdkDir = sdkDir
        self.activationWidth = activationWidth
        self.biasWidth = biasWidth
        self.weightWidth = weightWidth
        self.inputDimension = inputDimension
        self.outputDir = outputDir
        self.quantOverridesPath = quantOverridesPath
        self.config_file = config_file
        self.logger = logger
        self.__splitQuantizationVariations()

    def __splitQuantizationVariations(self):
        for quantVariation in self.quantizationVariations:
            quantList = quantVariation.split('_')
            uniqueQuantAlgos = set()
            if len(quantList) > 1:
                for quantAlgo in quantList[1:]:
                    uniqueQuantAlgos.add(quantAlgo)
            else:
                self.quantOptions.append(quantList[0])
        self.quantAlgorithms = [''] + list(uniqueQuantAlgos)
        # skip unquantized
        self.quantOptions = self.quantOptions[1:]

    def generate(self):
        configParams = extractEnvironmentConfigParams(os.path.abspath(self.config_file))
        mlFramework = ''
        if self.modelFile.endswith(".pb"):
            mlFramework = em.TENSORFLOW
        elif self.modelFile.endswith(".tflite"):
            mlFramework = em.TFLITE
        elif self.modelFile.endswith(".onnx"):
            mlFramework = em.ONNX
        else:
            self.logger.print("ERROR! Input model_file not recognizeable. Please use a model file with a .pb .onnx or .prototxt extension.")
            return -1
        em.setEnvironment(configParams, self.sdkDir, mlFramework)
        converter = None
        if mlFramework == em.TENSORFLOW:
            from qti.aisw.quantization_checker.utils.ConverterTools import TensorflowConverter
            converter = TensorflowConverter(self.logger, self.sdkDir, self.modelFile, self.inputList, self.inputDimension, self.quantOptions, self.quantAlgorithms)
        elif mlFramework == em.TFLITE:
            from qti.aisw.quantization_checker.utils.ConverterTools import TfliteConverter
            converter = TfliteConverter(self.logger, self.sdkDir, self.modelFile, self.inputList, self.inputDimension, self.quantOptions, self.quantAlgorithms)
        elif mlFramework == em.ONNX:
            from qti.aisw.quantization_checker.utils.ConverterTools import OnnxConverter
            converter = OnnxConverter(self.logger, self.sdkDir, self.modelFile, self.inputList, self.quantOptions, self.quantAlgorithms)
        else:
            self.logger.print("ERROR! Input model_file not recognizeable. Please use a model file with a .pb .onnx or .prototxt extension.")
            return -1
        try:
            if converter != None:
                environment = em.getEnvironment(configParams, self.sdkDir, mlFramework, sys.path[1])
                resultsMap = converter.convert(env=environment, activationWidth=self.activationWidth, biasWidth=self.biasWidth, weightWidth=self.weightWidth, workingDir=self.outputDir, quantOverrides=self.quantOverridesPath)
                for key, result in resultsMap.items():
                    if result == -1:
                        # conversion failure is not an error for the script, continue
                        self.logger.print('Error encountered during conversion of ' + key + ' quantization option. Please consult console/log output.')
                        return -1
        except Exception as e:
            self.logger.print("QOptionGenerator - ERROR! Conversion failed " + str(e))
            return -1
        return 0
