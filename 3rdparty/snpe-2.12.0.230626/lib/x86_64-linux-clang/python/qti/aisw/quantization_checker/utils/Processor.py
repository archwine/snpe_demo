#=============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from qti.aisw.quantization_checker.utils.Logger import Logger
from qti.aisw.quantization_checker.utils.Comparator import dequantizeBiases, compareBiases
from qti.aisw.quantization_checker.utils.Comparator import dequantizeWeights, compareWeights
from qti.aisw.quantization_checker.utils.Comparator import compareActivations, analyzeInputData

class Processor:
    def __init__(self, quantizationVariations, comparisonAlgorithms, logger : Logger):
        self.logger = logger
        self.quantizationVariations = quantizationVariations
        self.comparisonAlgorithms = comparisonAlgorithms

    def processWeightResults(self, opsMap):
        self.logger.print('Dequantizing weights...')
        dequantizeWeights(self.quantizationVariations, opsMap)
        self.logger.print('Comparing weights...')
        return compareWeights(self.quantizationVariations, opsMap, self.comparisonAlgorithms['weight_comparison_algorithms'], self.logger)

    def processBiasResults(self, opsMap):
        self.logger.print('Dequantizing biases...')
        dequantizeBiases(self.quantizationVariations, opsMap)
        self.logger.print('Comparing biases...')
        return compareBiases(self.quantizationVariations, opsMap, self.comparisonAlgorithms['bias_comparison_algorithms'], self.logger)

    def processActivationResults(self, opsMap):
        self.logger.print('Comparing activations...')
        return compareActivations(self.quantizationVariations, opsMap, self.comparisonAlgorithms['act_comparison_algorithms'])

    def processInputData(self, inputData):
        self.logger.print('Analyzing input data...')
        return analyzeInputData(inputData, self.comparisonAlgorithms['input_data_analysis_algorithms'])