#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from cmath import inf
from typing import Dict
import numpy as np
from qti.aisw.quantization_checker.utils.Logger import PrintOptions
from qti.aisw.quantization_checker.utils.Op import Op
import qti.aisw.quantization_checker.utils.Verifiers as Verifiers

class QuantizationComparisonAlgorithm:
    def __init__(self, algorithm : Verifiers.QcAlgorithm):
        self.algorithm = algorithm

    def compare(self, lhs, rhs, threshold):
        if self.algorithm == Verifiers.QcAlgorithm.MAX_ABS_DIFFERENCE:
            return MaxAbsDifference(threshold)(lhs, rhs)
        elif self.algorithm == Verifiers.QcAlgorithm.MIN_MAX_COMPARE:
            return MinMaxComparisonAlgorithm(threshold)(lhs, rhs)
        elif self.algorithm == Verifiers.QcAlgorithm.SQNR:
            return SqnrCalculation(threshold)(lhs, rhs)
        elif self.algorithm == Verifiers.QcAlgorithm.STATS:
            return CollectStats(threshold)(lhs)
        elif self.algorithm == Verifiers.QcAlgorithm.DATA_RANGE_CHECK:
            return BitWidthComparisonAlgorithm(threshold)(lhs)
        elif self.algorithm == Verifiers.QcAlgorithm.DATA_DISTRIBUTION_CHECK:
            return DistributionComparisonAlgorithm(threshold)(lhs)

class CollectStats:
    def __init__(self, threshold):
        self.threshold = float(threshold)

    def __call__(self, unquantizedData) -> Dict:
        median = np.median(unquantizedData)
        mean = np.mean(unquantizedData)
        variance = np.var(unquantizedData)
        stdDev = np.std(unquantizedData)
        min = np.amin(unquantizedData)
        max = np.amax(unquantizedData)
        skew = 0.0
        if stdDev != 0.0:
            skew = np.mean(np.power(((np.array(unquantizedData) - mean) / stdDev), 3))
        vals, counts = np.unique(unquantizedData, return_counts=True)
        index = np.argmax(counts)
        mode = vals[index]
        passes = True
        if abs(skew) > self.threshold:
            passes = False
        return {'pass': passes, 'data': 'skew: ' + str(skew) + ' min: ' + str(min) + ' max: ' + str(max) + ' median: ' + str(median) + ' variance: ' + str(variance) + ' stdDev: ' + str(stdDev) + ' mode: ' + str(mode), 'threshold': self.threshold}

class SqnrCalculation:
    def __init__(self, threshold):
        self.threshold = float(threshold)

    def __call__(self, unquantizedData, deQuantizedData) -> Dict:
        difference = np.subtract(deQuantizedData, unquantizedData)
        squaredDiff = np.square(difference)
        meanSquareError = np.mean(squaredDiff)

        if meanSquareError == 0:
            return {'pass': True, 'data': 'sqnr: ' + str(inf), 'threshold': self.threshold}
        else:
            squaredUnquantized = np.square(unquantizedData)
            meanSquareUnquantized = np.mean(squaredUnquantized)

            ratio = meanSquareUnquantized / meanSquareError
            logRatio = 10 * np.log10(ratio)
            passes = True
            if logRatio < self.threshold:
                passes = False
            return {'pass': passes, 'data': 'sqnr: ' + str(logRatio), 'threshold': self.threshold}

class MaxAbsDifference:
    def __init__(self, threshold):
        self.threshold = float(threshold)

    def __call__(self, lhs, rhs) -> Dict:
        result = np.absolute(np.subtract(lhs, rhs)).tolist()
        if result:
            maxResult = max(result)
            passes = True
            if maxResult > self.threshold:
                passes = False
            return {'pass': passes, 'data': 'largest absolute difference: ' + str(maxResult), 'threshold': self.threshold}
        else:
            return {'pass': True, 'data': 'largest absolute difference: ' + str(0.0), 'threshold': self.threshold}

class MinMaxComparisonAlgorithm:
    def __init__(self, threshold):
        self.threshold = float(threshold)

    def __call__(self, lhs, rhs) -> Dict:
        unquantizedMin = np.amin(lhs)
        unquantizedMax = np.amax(lhs)
        dequantizedMin = np.amin(rhs)
        dequantizedMax = np.amax(rhs)
        passes = True
        if abs(unquantizedMin - dequantizedMin) > self.threshold or abs(unquantizedMax - dequantizedMax) > self.threshold:
            passes = False
        return {'pass': passes, 'data': 'min: ' + str(abs(unquantizedMin - dequantizedMin)) + ' max: ' + str(abs(unquantizedMax - dequantizedMax)), 'threshold': self.threshold}

class BitWidthComparisonAlgorithm:
    def __init__(self, threshold):
        self.threshold = float(threshold)

    def __call__(self, data) -> Dict:
        uniques = np.unique(data.astype(int)).shape[0]
        dataRange = np.amax(data) - np.amin(data)
        passes = True
        if uniques > self.threshold or dataRange > self.threshold:
            passes = False
        return {'pass': passes, 'data': 'unique dec places: ' + str(uniques) + ' data range: ' + str(dataRange), 'threshold': self.threshold}

class DistributionComparisonAlgorithm:
    def __init__(self, threshold):
        self.thresholdPerBin = float(threshold)

    def __call__(self, data):
        dataWithPrecision = []
        for val in data:
            if abs(val - int(val)) > 0:
                dataWithPrecision.append(val)
        maxRatio = 0
        if len(dataWithPrecision) > 0:
            dataWithPrecision = np.array(dataWithPrecision)
            dataRange = int(max(dataWithPrecision) - min(dataWithPrecision) + 1)
            uint8_max = np.iinfo(np.uint8).max+1
            if (dataRange < uint8_max):
                dataRange = uint8_max
            hist, bins = np.histogram(dataWithPrecision, bins=dataRange)
            hist = np.where(hist > 0, hist-1, hist)
            distRatio = (hist/dataWithPrecision.shape)
            maxRatio = np.amax(distRatio)
        passes = True
        if maxRatio > self.thresholdPerBin:
            passes = False
        return {'pass': passes, 'data': 'Distribution of pixels above threshold: ' + str(maxRatio), 'threshold': self.thresholdPerBin}

class Comparator:
    def __init__(self, algorithm : Verifiers.QcAlgorithm):
        self.algorithm = algorithm

    def compare(self, lhs, rhs, threshold) -> Dict:
        qca = QuantizationComparisonAlgorithm(self.algorithm)
        return qca.compare(lhs, rhs, threshold)

def compareWeights(quantizationVariations, opsMap, comparisonAlgorithms, logger) -> Dict:
    setDataRangeAnalyzerThreshold(comparisonAlgorithms, Op.getWeightWidth())
    results = {}
    for quantizationVariation in quantizationVariations:
        # skip quantized models which fail to convert correctly
        if quantizationVariation not in opsMap:
            continue
        perOpResults = {}
        unquantizedWeights = None
        for opName in opsMap[quantizationVariation]:
            op = opsMap[quantizationVariation][opName]
            if op.getWeightName() not in (None, ''):
                # grab the corresponding unquantized op
                if quantizationVariation is not 'unquantized':
                    unquantizedOp = opsMap['unquantized'][opName]
                    unquantizedWeights = unquantizedOp.getWeights()
                else:
                    unquantizedWeights = op.getWeights()
                perOpResults[(opName, op.getWeightName())] = runLintingRules(quantizationVariation, unquantizedWeights, op.getDequantizedWeights(), op.getOpName(), op.getWeightName(), comparisonAlgorithms, logger)
        results[quantizationVariation] = perOpResults
    return results  

def comparePerOp(unquantizedData, dequantizedData, comparisonAlgorithms) -> Dict:
    results = {}
    if comparisonAlgorithms is not None:
        for comparisonAlgorithm in comparisonAlgorithms:
            results[comparisonAlgorithm['algo_name']] = doCompare(unquantizedData, dequantizedData, comparisonAlgorithm)
    else:
        results[Verifiers.MAX_DIFF] = doCompare(unquantizedData, dequantizedData, (Verifiers.MAX_DIFF, "0.5"))
    return results

def doCompare(unquantizedData, dequantizedData, comparisonAlgorithm) -> Dict:
    threshold = '0.0'
    comparator = None
    if dequantizedData is not None:
        if comparisonAlgorithm['algo_name'] == Verifiers.SQNR:
            comparator = Comparator(Verifiers.QcAlgorithm.SQNR)
            threshold = Verifiers.DEFAULT_THRESHOLDS.SQNR
        elif comparisonAlgorithm['algo_name'] == Verifiers.MAX_DIFF:
            comparator = Comparator(Verifiers.QcAlgorithm.MAX_ABS_DIFFERENCE)
            threshold = Verifiers.DEFAULT_THRESHOLDS.MAX_DIFF
        elif comparisonAlgorithm['algo_name'] == Verifiers.MIN_MAX:
            comparator = Comparator(Verifiers.QcAlgorithm.MIN_MAX_COMPARE)
            threshold = Verifiers.DEFAULT_THRESHOLDS.MIN_MAX
    else:
        if comparisonAlgorithm['algo_name'] == Verifiers.STATS:
            comparator = Comparator(Verifiers.QcAlgorithm.STATS)
            threshold = Verifiers.DEFAULT_THRESHOLDS.STATS
        elif comparisonAlgorithm['algo_name'] == Verifiers.DATA_DISTRIBUTION:
            comparator = Comparator(Verifiers.QcAlgorithm.DATA_DISTRIBUTION_CHECK)
            threshold = Verifiers.DEFAULT_THRESHOLDS.DATA_DISTRIBUTION
        elif comparisonAlgorithm['algo_name'] == Verifiers.DATA_RANGE:
            comparator = Comparator(Verifiers.QcAlgorithm.DATA_RANGE_CHECK)
    if 'threshold' in comparisonAlgorithm:
        threshold = comparisonAlgorithm['threshold']
    if comparator is not None:
        return comparator.compare(unquantizedData, dequantizedData, threshold)
    else:
        return {}

def dequantizeWeights(quantizationVariations, opsMap):
    for quantizationVariation in quantizationVariations[1:]:
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap:
            continue
        for item in opsMap[quantizationVariation].items():
            op = opsMap[quantizationVariation][item[0]]
            if op.getWeightName() not in (None, ''):
                dequantizedWeights = dequantizeOpWeights(op.getWeights(), op.getWeightsScaleOffset())
                op.setDequantizedWeights(dequantizedWeights)
                opsMap[quantizationVariation][item[0]] = op

def dequantizeOpWeights(quantizedWeights, weightsScaleOffset):
    scale = weightsScaleOffset['scale']
    offset = weightsScaleOffset['offset']
    return np.multiply(np.add(quantizedWeights, offset), scale)

def dequantizeBiases(quantizationVariations, opsMap):
    for quantizationVariation in quantizationVariations[1:]:
        # skip quantized models which are failed to get converted correctly
        if quantizationVariation not in opsMap:
            continue
        for item in opsMap[quantizationVariation].items():
            op = opsMap[quantizationVariation][item[0]]
            if op.getBiasName() not in (None, ''):
                # Nodes from PCQ models can have scale 0, we consider scale from weight/input nodes to dequantize those
                dequantizedBiases = dequantizeOpBiases(op.getBiases(), op.getBiasScale(), op.getBiasOffset())
                op.setDequantizedBiases(dequantizedBiases)
                opsMap[quantizationVariation][op.getOpName()] = op

def dequantizeOpBiases(quantizedBiases, biasScale, biasOffset):
    return np.multiply(np.add(quantizedBiases, biasOffset), biasScale)

def compareBiases(quantizationVariations, opsMap, comparisonAlgorithms, logger):
    setDataRangeAnalyzerThreshold(comparisonAlgorithms, Op.getBiasWidth())
    results = {}
    for quantizationVariation in quantizationVariations:
        # skip quantized models which fail to convert correctly
        if quantizationVariation not in opsMap:
            continue
        perOpResults = {}
        unquantizedBiases = None
        for opName in opsMap[quantizationVariation]:
            op = opsMap[quantizationVariation][opName]
            if op.getBiasName() not in (None, ''):
                # grab the corresponding unquantized op
                if quantizationVariation is not 'unquantized':
                    unquantizedOp = opsMap['unquantized'][opName]
                    unquantizedBiases = unquantizedOp.getBiases()
                else:
                    unquantizedBiases = op.getBiases()
                perOpResults[(opName, op.getBiasName())] = runLintingRules(quantizationVariation, unquantizedBiases, op.getDequantizedBiases(), op.getOpName(), op.getBiasName(), comparisonAlgorithms, logger)
        results[quantizationVariation] = perOpResults
    return results

def runLintingRules(quantizationVariation, unquantizedData, dequantizedData, opName, tensorName, comparisonAlgorithms, logger) -> Dict:
    dequantizedArray = None
    unquantizedArray = np.nan_to_num(np.array(unquantizedData))
    if quantizationVariation is not 'unquantized':
        dequantizedArray = np.nan_to_num(np.array(dequantizedData))
        dequantShape = list(dequantizedArray.shape)
        unquantShape = list(unquantizedArray.shape)
        if unquantShape != dequantShape:
            logger.print("WARNING! two data files have different shapes, " + str(unquantShape) + " vs " + str(dequantShape) + " please check manually! returning empty results", PrintOptions.LOGFILE)
            return {}
    return comparePerOp(unquantizedArray, dequantizedArray, comparisonAlgorithms)

def comparePerOpActivations(unquantizedActivations, quantizedScale, quantizedOffset, comparisonAlgorithms):
    setDataRangeAnalyzerThreshold(comparisonAlgorithms, Op.getActivationWidth())
    results = {}
    quantizedMin = 0
    quantizedMax = 0
    if quantizedScale is not None:
        quantizedMin = (0 + quantizedOffset) * quantizedScale
        quantizedMax = (int(Verifiers.getMaxValueBasedOnBitWidth(Op.getActivationWidth()))-1 + quantizedOffset) * quantizedScale
    if comparisonAlgorithms is not None:
        for comparisonAlgorithm in comparisonAlgorithms:                
            if quantizedScale is not None and comparisonAlgorithm['algo_name'] == "minmax":
                results[comparisonAlgorithm['algo_name']] = doActivationCompare(unquantizedActivations, (quantizedMin, quantizedMax), comparisonAlgorithm)
            elif comparisonAlgorithm['algo_name'] != "minmax":
                results[comparisonAlgorithm['algo_name']] = doActivationCompare(unquantizedActivations, None, comparisonAlgorithm)
    else:
        results['minmax'] = doActivationCompare(unquantizedActivations, (quantizedMin, quantizedMax), ("minmax", "0.5"))
    return results

def doActivationCompare(unquantizedActivations, minMax, comparisonAlgorithm):
    threshold = '0.0'
    if comparisonAlgorithm['algo_name'] == Verifiers.MIN_MAX:
        comparator = Comparator(Verifiers.QcAlgorithm.MIN_MAX_COMPARE)
        threshold = Verifiers.DEFAULT_THRESHOLDS.MIN_MAX
    elif comparisonAlgorithm['algo_name'] == Verifiers.STATS:
        comparator = Comparator(Verifiers.QcAlgorithm.STATS)
        threshold = Verifiers.DEFAULT_THRESHOLDS.STATS
    elif comparisonAlgorithm['algo_name'] == "data_range_analyzer":
        comparator = Comparator(Verifiers.QcAlgorithm.DATA_RANGE_CHECK)
    else:
        comparator = Comparator(Verifiers.QcAlgorithm.MIN_MAX_COMPARE)
        threshold = Verifiers.DEFAULT_THRESHOLDS.MIN_MAX
    if 'threshold' in comparisonAlgorithm:
        threshold = comparisonAlgorithm['threshold']
    return comparator.compare(unquantizedActivations, minMax, threshold)

def compareActivations(quantizationVariations, opsMap, comparisonAlgorithms):
    results = {}
    unquantizedOps = opsMap['unquantized']
    for quantizationVariation in quantizationVariations:
        # skip quantized models which fail to convert
        if quantizationVariation not in opsMap:
            continue
        perOpResults = {}
        for opName in opsMap[quantizationVariation]:
            quantizedOp = opsMap[quantizationVariation][opName]
            for unquantOpName in unquantizedOps:
                unquantizedOp = opsMap['unquantized'][unquantOpName]
                if quantizedOp.getActivationNodeName() not in (None, '') and quantizedOp.getActivationNodeName() == unquantizedOp.getActivationNodeName():
                    activationList = unquantizedOp.getActivations()
                    perInputResults = {}
                    for activationPerInput in activationList:
                        if quantizationVariation == 'unquantized':
                            perInputResults[activationPerInput[0]] = comparePerOpActivations(activationPerInput[1], None, None, comparisonAlgorithms)
                        else:
                            perInputResults[activationPerInput[0]] = comparePerOpActivations(activationPerInput[1], quantizedOp.getActivationScale(), quantizedOp.getActivationOffset(), comparisonAlgorithms)
                    perOpResults[(opName, quantizedOp.getActivationNodeName())] = perInputResults
                    break
        results[quantizationVariation] = perOpResults
    return results

def analyzeInputData(inputData, comparisonAlgorithms):
    results = {}
    for filename in inputData:
        results[filename] = analyzeInput(inputData[filename], comparisonAlgorithms)
    return results

def analyzeInput(inputTensor, comparisonAlgorithms):
    results = {}
    if comparisonAlgorithms is not None:
        for comparisonAlgorithm in comparisonAlgorithms:
            results[comparisonAlgorithm['algo_name']] = doInputAnalysis(inputTensor, comparisonAlgorithm)
    else:
        results[Verifiers.STATS] = doInputAnalysis(inputTensor, (Verifiers.STATS, Verifiers.DEFAULT_THRESHOLDS.STATS))
    return results

def doInputAnalysis(inputTensor, comparisonAlgorithm):
    if comparisonAlgorithm['algo_name'] == Verifiers.STATS:
        comparator = Comparator(Verifiers.QcAlgorithm.STATS)
    return comparator.compare(inputTensor, None, threshold=Verifiers.DEFAULT_THRESHOLDS.STATS)

def setDataRangeAnalyzerThreshold(comparisonAlgorithms, bitWidth):
    for comparisonAlgorithm in comparisonAlgorithms:
        if comparisonAlgorithm['algo_name'] == Verifiers.DATA_RANGE:
            comparisonAlgorithm['threshold'] = Verifiers.getMaxValueBasedOnBitWidth(bitWidth)
