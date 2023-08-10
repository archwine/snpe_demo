#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import logging
from qti.aisw.quantization_checker.utils.Logger import Logger, PrintOptions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from qti.aisw.quantization_checker.utils import utils
from qti.aisw.quantization_checker.utils.Progress import Progress, ProgressStage
import qti.aisw.quantization_checker.utils.Constants as Constants

class TensorflowConverter:
    def __init__(self, logger: Logger, sdkPath, inputNetwork, inputList, inputDimension, quantizationVariations=None, quantizationAlgorithms=None):
        logging.disable(logging.WARNING)
        self.tf = __import__('tensorflow')
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__inputList = inputList
        self.__snpeTfQuantizerArgs = {}
        inputsAndShapes, outputNames = self.__getTfGraphInputsAndOutputs__()
        self.__inputArgs = inputsAndShapes
        self.__outputArgs = outputNames
        self.logger = logger
        self.__buildArgs__(quantizationVariations, quantizationAlgorithms)
        self.__inputDimension = inputDimension

    def convert(self, env, activationWidth=None, biasWidth=None, weightWidth=None, workingDir=None, quantOverrides=None):
        snpeTensorflowConverterBinaryPath = os.path.join(self.__sdkPath, Constants.BIN_PATH_IN_SDK, Constants.TF_CONVERTER_BIN_NAME)

        if not workingDir:
            workingDir = os.path.dirname(self.__inputNetwork)
        utils.changeDir(workingDir)

        if self.__inputDimension: inputArgsWithSwitches = ' -d '.join(self.__inputDimension)
        else: inputArgsWithSwitches = ' -d '.join(self.__inputArgs)

        outputArgsWithSwitches = ' --out_node '.join(self.__outputArgs)
        baseArgs = ' -d ' + inputArgsWithSwitches + ' --out_node ' + outputArgsWithSwitches + ' -i ' + self.__inputNetwork
        unquantizedDlc = os.path.join(workingDir, 'unquantized', 'unquantized') + '.dlc'
        outArgs = ' -o ' + unquantizedDlc
        self.logger.print('Converting TF model to SNPE DLC', PrintOptions.LOGFILE)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

        returnValue = utils.issueCommandAndWait(snpeTensorflowConverterBinaryPath + baseArgs + outArgs, self.logger, env)
        if returnValue != 0:
            return ('unquantized', -1)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

        resultsMap = {}
        resultsMap = snpeQuantizeDlc(self.__snpeTfQuantizerArgs, unquantizedDlc, self.__inputNetwork, self.__sdkPath, env, self.logger, activationWidth, biasWidth, weightWidth, quantOverrides)
        return resultsMap

    def __buildArgs__(self, quantizationVariations=None, quantizationAlgorithms=None):
        self.logger.print('Input list: ' + self.__inputList, PrintOptions.LOGFILE)
        self.__snpeTfQuantizerArgs = buildQuantizationParameterMap(quantizationVariations, quantizationAlgorithms, self.__inputList)

    def __getTfGraphInputsNameAndShape__(self, graph_def):
        inputTensors = []
        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name='')
        for op in graph.get_operations():
            if op.type == "Placeholder":
                for output in op.outputs:
                    if output.get_shape().is_fully_defined():
                        inputTensors.append([op.name, output.get_shape().as_list()])
                    else:
                        inputTensors.append([op.name, [None]])


        inputsAndShapes = []
        for inputTensor in inputTensors:
            if None in inputTensor[1]:
                inputTensor = promptUserForInputDims(inputTensor)
            listToStr = ','.join(map(str, inputTensor[1]))
            inputsAndShapes.append(inputTensor[0] + ' ' + listToStr)

        return inputsAndShapes

    def __getTfGraphInputsAndOutputs__(self):
        tfGraph = self.__getTfGraph__(self.__inputNetwork)
        inputsAndShapes = self.__getTfGraphInputsNameAndShape__(tfGraph)
        outputNames = self.__getTfGraphOutputsName__(tfGraph)
        return (inputsAndShapes, outputNames)

    def __getTfGraphOutputsName__(self, graph_def):
        outputs = []
        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name='')
            ops = self.tf.compat.v1.get_default_graph().get_operations()
            outputs_set = set(ops)
            for op in ops:
                if len(op.inputs) == 0 and op.type != 'Const':#network input nodes detected
                    continue
                else:
                    for input_tensor in op.inputs:
                        if input_tensor.op in outputs_set:
                            outputs_set.remove(input_tensor.op)

        for op in outputs_set:
            outputs.append(op.node_def.name)
        return outputs

    def __getTfGraph__(self, pbFile):
        session = self.tf.compat.v1.Session(graph=self.tf.Graph())
        with session.graph.as_default():
            graph_def = self.tf.compat.v1.GraphDef()
            with open(pbFile, "rb") as f:
                graph_def.ParseFromString(f.read())
            self.tf.import_graph_def(graph_def, name="")
        return graph_def

class TfliteConverter:
    def __init__(self, logger: Logger, sdkPath, inputNetwork, inputList, inputDimension, quantizationVariations=None, quantizationAlgorithms=None):
        logging.disable(logging.WARNING)
        self.tf = __import__('tensorflow')
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__inputList = inputList
        self.__snpeTfliteQuantizerArgs = {}
        inputsAndShapes, outputNames = self.__getTfliteGraphInputsAndOutputs__()
        self.__inputArgs = inputsAndShapes
        self.__outputArgs = outputNames
        self.logger = logger
        self.__buildArgs__(quantizationVariations, quantizationAlgorithms)
        self.__inputDimension= inputDimension

    def convert(self, env, activationWidth=None, biasWidth=None, weightWidth=None, workingDir=None, quantOverrides=None):
        snpeTfliteConverterBinaryPath = os.path.join(self.__sdkPath, Constants.BIN_PATH_IN_SDK, Constants.TFLITE_CONVERTER_BIN_NAME)

        if not workingDir:
            workingDir = os.path.dirname(self.__inputNetwork)
        utils.changeDir(workingDir)

        if self.__inputDimension: inputArgsWithSwitches = ' -d '.join(self.__inputDimension)
        else: inputArgsWithSwitches = ' -d '.join(self.__inputArgs)

        outputArgsWithSwitches = ' --out_node '.join(self.__outputArgs)
        baseArgs = ' -d ' + inputArgsWithSwitches + ' --out_node ' + outputArgsWithSwitches + ' -i ' + self.__inputNetwork
        unquantizedDlc = os.path.join(workingDir, 'unquantized', 'unquantized') + '.dlc'
        outArgs = ' -o ' + unquantizedDlc
        self.logger.print('Converting TFLite model to SNPE DLC', PrintOptions.LOGFILE)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

        returnValue = utils.issueCommandAndWait(snpeTfliteConverterBinaryPath + baseArgs + outArgs, self.logger, env)
        if returnValue != 0:
            return ('unquantized', -1)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

        resultsMap = {}
        resultsMap = snpeQuantizeDlc(self.__snpeTfliteQuantizerArgs, unquantizedDlc, self.__inputNetwork, self.__sdkPath, env, self.logger, activationWidth, biasWidth, weightWidth, quantOverrides)
        return resultsMap

    def __buildArgs__(self, quantizationVariations=None, quantizationAlgorithms=None):
        self.logger.print('Input list: ' + self.__inputList, PrintOptions.LOGFILE)
        self.__snpeTfliteQuantizerArgs = buildQuantizationParameterMap(quantizationVariations, quantizationAlgorithms, self.__inputList)

    def __getTfliteGraphInputsNameAndShape__(self, graph_def):
        inputsAndShapes = []
        interpreter = self.tf.lite.Interpreter(model_content=graph_def)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()
        for iter_info in input_info:
            dims = iter_info['shape']
            inputShape = []
            for dim in dims:
                inputShape.append(dim)
            listToStr = ','.join(map(str, inputShape))
            inputsAndShapes.append(iter_info['name'] + ' '  + listToStr)
        return inputsAndShapes

    def __getTfliteGraphInputsAndOutputs__(self):
        tfliteGraph = self.__getTfliteGraph__(self.__inputNetwork)
        inputsAndShapes = self.__getTfliteGraphInputsNameAndShape__(tfliteGraph)
        outputNames = self.__getTfliteGraphOutputsName__(tfliteGraph)
        return (inputsAndShapes, outputNames)

    def __getTfliteGraphOutputsName__(self, graph_def):
        outputs = []
        interpreter = self.tf.lite.Interpreter(model_content=graph_def)
        interpreter.allocate_tensors()
        output_info = interpreter.get_output_details()
        for iter_info in output_info:
            outputs.append(iter_info['name'])
        return outputs

    def __getTfliteGraph__(self, tfliteFile):
        with open(tfliteFile, 'rb') as fid:
            tfliteGraph = fid.read()
        return tfliteGraph

class OnnxConverter:
    def __init__(self, logger: Logger, sdkPath, inputNetwork, inputList, quantizationVariations=None, quantizationAlgorithms=None):
        self.__sdkPath = sdkPath
        self.__inputNetwork = inputNetwork
        self.__inputList = inputList
        self.__snpeOnnxQuantizerArgs = {}
        self.logger = logger
        self.__buildArgs__(quantizationVariations, quantizationAlgorithms)

    def convert(self, env, activationWidth=None, biasWidth=None, weightWidth=None, workingDir=None, quantOverrides=None):
        snpeOnnxConverterBinaryPath = os.path.join(self.__sdkPath, Constants.BIN_PATH_IN_SDK, Constants.ONNX_CONVERTER_BIN_NAME)

        if not workingDir:
            workingDir = os.path.dirname(self.__inputNetwork)
        utils.changeDir(workingDir)
        
        baseArgs = ' -i ' + self.__inputNetwork
        unquantizedDlc = os.path.join(workingDir, 'unquantized', 'unquantized') + '.dlc'
        outArgs = ' -o ' + unquantizedDlc
        self.logger.print('Converting ONNX model to SNPE DLC', PrintOptions.LOGFILE)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

        returnValue = utils.issueCommandAndWait(snpeOnnxConverterBinaryPath + baseArgs + outArgs, self.logger, env)
        if returnValue != 0:
            return ('unquantized', -1)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

        resultsMap = {}
        resultsMap = snpeQuantizeDlc(self.__snpeOnnxQuantizerArgs, unquantizedDlc, self.__inputNetwork, self.__sdkPath, env, self.logger, activationWidth, biasWidth, weightWidth, quantOverrides)
        return resultsMap

    def __buildArgs__(self, quantizationVariations=None, quantizationAlgorithms=None):
        self.logger.print('Input list: ' + self.__inputList, PrintOptions.LOGFILE)
        self.__snpeOnnxQuantizerArgs = buildQuantizationParameterMap(quantizationVariations, quantizationAlgorithms, self.__inputList)

    def __getOnnxGraphInputs__(self):
        import onnx
        model = onnx.load(self.__inputNetwork)

        parameterNames = set()
        for tensor in model.graph.initializer:
            parameterNames.add(str(tensor.name))

        inputTensors = []
        for input in model.graph.input:
            inputInfo = []
            name = str(input.name)
            if name in parameterNames:
                continue
            dims = []
            tensorType = input.type.tensor_type
            if (tensorType.HasField("shape")):
                for dim in tensorType.shape.dim:
                    if (dim.HasField("dim_value")):
                        dims.append(dim.dim_value)
                    elif (dim.HasField("dim_param")):
                        dims.append(dim.dim_param)
                    else:
                        dims.append('?')
            else:
                self.logger.print("ERROR: Unknown input shape", PrintOptions.LOGFILE)
            inputInfo = [input.name, dims]
            if not self.__checkOnnxForUnknownDims__(dims):
                inputInfo = promptUserForInputDims(self.logger, inputInfo)
            listToStr = ','.join(map(str, inputInfo[1]))
            inputTensors.append(inputInfo[0] + ' ' + listToStr)

        return inputTensors

    def __checkOnnxForUnknownDims__(self, dims):
        return all(isinstance(dim, int) for dim in dims)

def promptUserForInputDims(logger, inputTensor):
    logger.print('Input found with unknown dimensions...')
    logger.print('Please enter the dimensions for the following input: ' + inputTensor[0] + ' ' + ','.join(map(str, inputTensor[1])) + ', in the format B,H,W,C: ')
    dimensions = input()
    strToList = dimensions.split(',')
    if len(strToList) != 4:
        logger.print('Error parsing input dimensions. Exiting generator tool.', PrintOptions.LOGFILE)
        exit()
    inputTensor[1] = strToList
    return inputTensor

def buildQuantizationParameterMap(quantizationVariations, quantizationAlgorithms, inputList):
    parameterMap = {}
    quantizationVariationsMasterMap = {'tf': '', 'enhanced': '--use_enhanced_quantizer', 'adjusted': '--use_adjusted_weights_quantizer', 'symmetric': '--use_symmetric_quantize_weights'}
    quantizationAlgorithmsMasterMap = {'': '', 'cle': '--optimizations cle', 'bc': '--optimizations bc',  'cle_bc': '--optimizations cle --optimizations bc'}
    quantizationVariationsMap = {}
    quantizationAlgorithmsMap = {}
    if quantizationVariations:
        for quantVariation in quantizationVariations:
            quantizationVariationsMap[quantVariation] = quantizationVariationsMasterMap[quantVariation]
    else:
        quantizationVariationsMap = quantizationVariationsMasterMap
    if quantizationAlgorithms:
        for quantAlgorithm in quantizationAlgorithms:
            quantizationAlgorithmsMap[quantAlgorithm] = quantizationAlgorithmsMasterMap[quantAlgorithm]
    else:
        quantizationAlgorithmsMap = quantizationAlgorithmsMasterMap

    inputListArg = ' --input_list=' + inputList
    for quantizationVariation in quantizationVariationsMap.keys():
        for quantizationAlgorithm in quantizationAlgorithmsMap.keys():
            parameterMap[quantizationVariation + ('_' + quantizationAlgorithm if quantizationAlgorithm else '')] = inputListArg + (' ' + quantizationVariationsMap[quantizationVariation] if quantizationVariationsMap[quantizationVariation] else '') + (' ' + quantizationAlgorithmsMap[quantizationAlgorithm] if quantizationAlgorithm else '')

    return parameterMap

def snpeQuantizeDlc(snpeDlcQuantizeArgs, unquantizedDlc, inputNetwork, sdkPath, env, logger, activationBitwidth=None, biasBitwidth=None, weightBitwidth=None, quantOverrides=None):
    resultsMap = {}
    snpeDlcQuantizeBinaryPath = os.path.join(sdkPath, Constants.BIN_PATH_IN_SDK, Constants.QUANTIZER_BIN_NAME)
    workingDir = os.path.dirname(inputNetwork)
    for paramName, params in snpeDlcQuantizeArgs.items():
        utils.makeSubdir(paramName)
        quantizerBaseArgs = ' --input_dlc=' + unquantizedDlc + params
        if activationBitwidth != None:
            quantizerBaseArgs += ' --act_bitwidth=' + activationBitwidth
        if biasBitwidth != None:
            quantizerBaseArgs += ' --bias_bitwidth=' + biasBitwidth
        if weightBitwidth != None:
            quantizerBaseArgs += ' --weights_bitwidth=' + weightBitwidth
        if quantOverrides != None:
            quantizerBaseArgs += ' --override_params ' + quantOverrides
        outArgs = ' --output_dlc=' + os.path.join(workingDir, paramName, paramName + '.dlc')
        resultsMap[paramName] = utils.issueCommandAndWait(snpeDlcQuantizeBinaryPath + quantizerBaseArgs + outArgs, logger, env, False)
        Progress.updateProgress(Progress.getStepSize(ProgressStage.GENERATOR))

    return resultsMap
