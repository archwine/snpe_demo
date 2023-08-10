# =============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from enum import Enum

class HistogramGeneration(Enum):
    SKIP_GENERATION = 1
    GENERATE_HISTOGRAM = 2
    GENERATE_PER_CHANNEL_HISTOGRAM = 3

class HistogramVisualizer:
    @staticmethod
    def visualize(unquantized_data, quantized_data, dest):
        fig, (ax_1, ax_2) = plt.subplots(1, 2, sharex='col', sharey='row')
        ax_1.set_xlabel('Quantized Tensor Value Range')
        ax_1.set_ylabel('Frequency')
        ax_2.set_xlabel('Unquantized Tensor Value Range')

        ax_1.hist(quantized_data)
        ax_2.hist(unquantized_data, color='gold')
        plt.tight_layout()
        plt.savefig(dest)
        plt.close(fig)

def visualizeWeightTensors(quantizationVariations, opsMap, hist_analysis_dir, logger):
    unquantizedOps = opsMap['unquantized']
    for quantizationVariation in quantizationVariations[1:]:
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap:
            continue
        for opName in opsMap[quantizationVariation]:
            quantizedOp = opsMap[quantizationVariation][opName]
            if quantizedOp.getWeightName() not in (None, ''):
                histAnalysisPerQuantPerOpDir = os.path.join(hist_analysis_dir, quantizationVariation)
                if not os.path.exists(histAnalysisPerQuantPerOpDir):
                    os.makedirs(histAnalysisPerQuantPerOpDir)
                logger.print('Processing Weight Histogram Distribution for OP - ' + opName + ' from Quant option - ' + quantizationVariation)
                unquantizedWeights = unquantizedOps[opName].getWeights()
                quantizedWeights = quantizedOp.getWeights()
                weightsDims = quantizedOp.getWeightsDims()
                unquantizedWeights = unquantizedWeights.reshape(weightsDims)
                quantizedWeights = quantizedWeights.reshape(weightsDims)
                sanitizedOpName = opName.replace('/', '_').strip()
                savePath = os.path.join(histAnalysisPerQuantPerOpDir, sanitizedOpName)
                HistogramVisualizer.visualize(unquantizedWeights[:].flatten(), quantizedWeights[:].flatten(), savePath + ".png")

def visualizeBiasTensors(quantizationVariations, opsMap, hist_analysis_dir, logger):
    unquantizedOps = opsMap['unquantized']
    for quantizationVariation in quantizationVariations[1:]:
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap:
            continue
        histAnalysisPerQuantDir = os.path.join(hist_analysis_dir, quantizationVariation)
        for opName in opsMap[quantizationVariation]:
            quantizedOp = opsMap[quantizationVariation][opName]
            if quantizedOp.getBiasName() not in (None, ''):
                if not os.path.exists(histAnalysisPerQuantDir):
                    os.makedirs(histAnalysisPerQuantDir)
                logger.print('Processing Bias Histogram Distribution for OP - ' + opName + ' from Quant option - ' + quantizationVariation)
                unquantizedBiases = unquantizedOps[quantizedOp.getOpName()].getBiases()
                quantizedBiases = quantizedOp.getBiases()
                sanitizedOpName = opName.replace('/', '_').strip()
                savePath = os.path.join(histAnalysisPerQuantDir, sanitizedOpName)
                HistogramVisualizer.visualize(unquantizedBiases.flatten(), quantizedBiases.flatten(), savePath + ".png")
