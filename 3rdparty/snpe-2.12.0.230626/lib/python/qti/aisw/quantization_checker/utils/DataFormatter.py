#=============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import os
import re
import sys

from qti.aisw.quantization_checker.utils.ResultNode import ResultNode
from qti.aisw.quantization_checker.utils.utils import *
from qti.aisw.quantization_checker.utils.Logger import Logger, PrintOptions, NEWLINE
from qti.aisw.quantization_checker.utils.Table import Table, Row, Cell
import qti.aisw.quantization_checker.utils.Verifiers as Verifiers

class DataFormatter:
    def __init__(self, outputDir, inputModel, inputList, quantizationVariations, logger=Logger('qnn-quantization-checker-log')):
        self.__PASSDIV = '<div class=pass>Pass</div>'
        self.__FAILDIV = '<div class=fail>Fail</div>'
        self.outputDir = outputDir
        self.inputModel = inputModel
        self.inputList = inputList
        self.quantizationVariations = quantizationVariations
        self.logger = logger
        self.inputResults = {}
        self.activationResults = {}
        self.weightResults = {}
        self.biasResults = {}
        self.inputsWeightsAndBiasesAnalysisCsvHeader = ['Node Name', 'Tensor Name', 'Passes Verification', 'Threshold for Verification']
        self.activationsAnalysisCsvHeader = self.inputsWeightsAndBiasesAnalysisCsvHeader + ['Input Filename']
        self.htmlHeadBegin = '''
            <html>
            <head><title>{title}</title>
        '''
        self.htmlHeadEnd = '''
            <style>
                h1 {
                    font-family: Arial;
                }
                h2 {
                    font-family: Arial;
                }
                h3 {
                    font-family: Arial;
                }
                #nodes table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: auto !important;
                }
                #top table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: auto !important;
                }
                div table {
                    font-size: 11pt;
                    font-family: Arial;
                    border-collapse: collapse;
                    border: 1px solid silver;
                    table-layout: auto !important;
                }

                #legend td, th {
                    padding: 5px;
                    width: auto !important;
                    white-space:nowrap;
                }
                #nodes td, th {
                    padding: 5px;
                    width: auto !important;
                    white-space:nowrap;
                }

                #legend tr:nth-child(even) {
                    background: #E0E0E0;
                }
                #nodes tr:nth-child(even) {
                    background: #E0E0E0;
                }

                #nodes tr:hover {
                    background: silver;
                }
                #nodes tr.th:hover {
                    background: silver;
                }
                .pass {
                    color: green;
                }
                .fail {
                    color: red;
                    font-weight: bold;
                }
            </style>
            </head>
        '''
        self.htmlBody = '''
            <body>
                <div>
                    <h1>{title}</h1>
                </div>
                <div>{legend}</div>
                <br/>
                <div>
                    <h2>{summaryTitle}</h2>
                    <h3>{instructions}</h3>
                </div>
                <div id=nodes>{summary}</div>
                <br/>
                <h2>{allNodesTitle}</h2>
                <div id=nodes>{table}</div>
            </body>
            </html>
        '''

    def setInputResults(self, inputResults):
        self.inputResults = inputResults

    def setActivationsResults(self, activationResults):
        self.activationResults = activationResults

    def setWeightResults(self, weightResults):
        self.weightResults = weightResults

    def setBiasResults(self, biasResults):
        self.biasResults = biasResults

    def __formatDataForHtml(self) -> Dict:
        results = {}
        for quantizationVariation in self.quantizationVariations:
            resultsPerInput = {}
            for inputFile in self.inputList:
                quantOptionResults = []
                if quantizationVariation in self.weightResults:
                    for opAndNodeName, comparisonResults in self.weightResults[quantizationVariation].items():
                        comparisonPassFails = {}
                        for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                            if comparisonResult != dict():
                                comparisonPassFails[comparisonAlgorithmName] = self.__translateTrueFalseToPassFail(comparisonResult['pass'])
                        quantOptionResults.append(ResultNode(opAndNodeName[0], opAndNodeName[1], 'Weight', comparisonPassFails))
                if quantizationVariation in self.biasResults:
                    for opAndNodeName, comparisonResults in self.biasResults[quantizationVariation].items():
                        comparisonPassFails = {}
                        for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                            if comparisonResult != dict():
                                comparisonPassFails[comparisonAlgorithmName] = self.__translateTrueFalseToPassFail(comparisonResult['pass'])
                        quantOptionResults.append(ResultNode(opAndNodeName[0], opAndNodeName[1], 'Bias', comparisonPassFails))
                if quantizationVariation != 'unquantized' and quantizationVariation in self.activationResults:
                    for opAndNodeName, inputFileResult in self.activationResults[quantizationVariation].items():
                        if inputFileResult.get(inputFile) is not None:
                            comparisonResults = inputFileResult[inputFile]
                            comparisonPassFails = {}
                            for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                                if comparisonResult != dict():
                                    comparisonPassFails[comparisonAlgorithmName] = self.__translateTrueFalseToPassFail(comparisonResult['pass'])
                            quantOptionResults.append(ResultNode(opAndNodeName[0], opAndNodeName[1], 'Activation', comparisonPassFails))
                resultsPerInput[inputFile] = quantOptionResults
            results[quantizationVariation] = resultsPerInput
        inputAnalysisResults = []
        for filename in self.inputResults:
            comparisonPassFails = {}
            for comparisonAlgorithmName, comparisonResult in self.inputResults[filename].items():
                if comparisonResult != dict():
                    comparisonPassFails[comparisonAlgorithmName] = self.__translateTrueFalseToPassFail(comparisonResult['pass'])
            inputAnalysisResults.append(ResultNode(filename, "N/A", "N/A", comparisonPassFails))
        results["inputData"] = inputAnalysisResults
        return results

    def __translateTrueFalseToPassFail(self, result):
        if result:
            return self.__PASSDIV
        else:
            return self.__FAILDIV

    def __getAlgorithmNames(self, results) -> List:
        algorithmsWithDups = []
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation not in results:
                continue
            for filename, comparisonResults in results[quantizationVariation].items():
                for comparisonResult in comparisonResults:
                    algorithmsWithDups.extend(comparisonResult.getResults().keys())
        return list(dict.fromkeys(algorithmsWithDups))

    def __getInputAnalysisAlgoNames(self) -> List:
        algorithmNames = []
        for filename in self.inputResults:
            for comparisonAlgorithmName, comparisonResult in self.inputResults[filename].items():
                if comparisonAlgorithmName not in algorithmNames:
                    algorithmNames.append(comparisonAlgorithmName)
        return algorithmNames

    # TODO: move to Comparator so as not to duplicate code and names
    def __translateAlgoNames(self, algorithms):
        readableAlgoNames = []
        for algorithm in algorithms:
            if algorithm == Verifiers.STATS:
                readableAlgoNames.append(Verifiers.STATS_DESCRIPTIVE_NAME)
            elif algorithm == Verifiers.DATA_RANGE:
                readableAlgoNames.append(Verifiers.DATA_RANGE_DESCRIPTIVE_NAME)
            elif algorithm == Verifiers.SQNR:
                readableAlgoNames.append(Verifiers.SQNR_DESCRIPTIVE_NAME)
            elif algorithm == Verifiers.MIN_MAX:
                readableAlgoNames.append(Verifiers.MIN_MAX_DESCRIPTIVE_NAME)
            elif algorithm == Verifiers.MAX_DIFF:
                readableAlgoNames.append(Verifiers.MAX_DIFF_DESCRIPTIVE_NAME)
            elif algorithm == Verifiers.DATA_DISTRIBUTION:
                readableAlgoNames.append(Verifiers.DATA_DISTRIBUTION_DESCRIPTIVE_NAME)
        return readableAlgoNames

    def __getMultiIndexForColumns(self, algorithms):
        return pd.MultiIndex.from_product([list(['Quantization Checker Pass/Fail']), list(algorithms)])

    def __getResultOnlyAlgos(self, results, algorithmKeys) -> List:
        onlyAlgosList = []
        for result in results:
            valuesList = []
            for key in algorithmKeys:
                if key in result.getResults().keys():
                    valuesList.append(result.getResults()[key])
                else:
                    valuesList.append('N/A')
            valuesList.append(result.getAnalysisDescription())
            onlyAlgosList.append(valuesList)
        return onlyAlgosList

    def __getResultWithoutAlgos(self, results) -> List:
        noVerifiers = []
        for result in results:
            noVerifiers.append([result.getNodeName(), result.getTensorName(), result.getTensorType()])
        return noVerifiers

    def __generateAlgorithmDescriptions(self, readableAlgorithmNames):
        descriptions = {}
        for readableAlgorithmName in readableAlgorithmNames:
            if readableAlgorithmName == Verifiers.STATS_DESCRIPTIVE_NAME:
                descriptions[readableAlgorithmName] = Verifiers.STATS_DESCRIPTION
            elif readableAlgorithmName == Verifiers.DATA_RANGE_DESCRIPTIVE_NAME:
                descriptions[readableAlgorithmName] = Verifiers.DATA_RANGE_DESCRIPTION
            elif readableAlgorithmName == Verifiers.SQNR_DESCRIPTIVE_NAME:
                descriptions[readableAlgorithmName] = Verifiers.SQNR_DESCRIPTION
            elif readableAlgorithmName == Verifiers.MIN_MAX_DESCRIPTIVE_NAME:
                descriptions[readableAlgorithmName] = Verifiers.MIN_MAX_DESCRIPTION
            elif readableAlgorithmName == Verifiers.MAX_DIFF_DESCRIPTIVE_NAME:
                descriptions[readableAlgorithmName] = Verifiers.MAX_DIFF_DESCRIPTION
            elif readableAlgorithmName == Verifiers.DATA_DISTRIBUTION_DESCRIPTIVE_NAME:
                descriptions[readableAlgorithmName] = Verifiers.DATA_DISTRIBUTION_DESCRIPTION
        return descriptions

    def __getLegendTable(self, algorithms, descriptions):
        htmlString = '<table id=legend><tr align=left><th colspan=2>Legend:</th></tr><tr><th>Quantization Checker Name</th><th>Description</th></tr>'
        # skip the description column since it is self explanatory and is itself an explanation
        for algorithm in algorithms[:-1]:
            htmlString += '<tr><td>' + algorithm + '</td><td>' + descriptions[algorithm] + '</td></tr>'
        htmlString += '</table>'
        return htmlString

    def printHtml(self):
        results = self.__formatDataForHtml()
        algorithmNames = self.__getAlgorithmNames(results)
        readableAlgorithmNames = self.__translateAlgoNames(algorithmNames)
        algoDescriptions = self.__generateAlgorithmDescriptions(readableAlgorithmNames)
        # add the description column manually since we need to interpret the results at the end
        readableAlgorithmNames.append('Description')
        columns = self.__getMultiIndexForColumns(readableAlgorithmNames)
        htmlOutputDir = ''
        max_colwidth = None
        if sys.version_info < (3, 8):
            # display.max_colwidth only uses an int argument prior to 3.8
            max_colwidth = -1
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation not in results:
                continue
            for inputFile in self.inputList:
                noAlgorithms = self.__getResultWithoutAlgos(results[quantizationVariation][inputFile])
                onlyAlgorithms = self.__getResultOnlyAlgos(results[quantizationVariation][inputFile], algorithmNames)
                failedNodes = self.__getFailedNodes(results[quantizationVariation][inputFile])
                noAlgorithmsFailed = self.__getResultWithoutAlgos(failedNodes)
                onlyAlgorithmsFailed = self.__getResultOnlyAlgos(failedNodes, algorithmNames)
                pd.set_option('display.max_colwidth', max_colwidth)

                dfOnlyAlgos = pd.DataFrame(onlyAlgorithms, columns=columns)
                dfNoAlgos = pd.DataFrame(noAlgorithms, columns=pd.MultiIndex.from_product([list(['']), list(['Op Name', 'Node Name', 'Node Type'])]))
                dfResults = pd.concat([dfNoAlgos, dfOnlyAlgos], axis=1)
                dfResults = dfResults.set_index([('', 'Op Name'), ('', 'Node Name')], drop=True)
                dfResults.index.names = ['Op Name', 'Node Name']
                dfResults.sort_index(inplace=True)

                dfOnlyAlgosFailed = pd.DataFrame(onlyAlgorithmsFailed, columns=columns)
                dfNoAlgosFailed = pd.DataFrame(noAlgorithmsFailed, columns=pd.MultiIndex.from_product([list(['']), list(['Op Name', 'Node Name', 'Node Type'])]))
                dfResultsFailed = pd.concat([dfNoAlgosFailed, dfOnlyAlgosFailed], axis=1)
                dfResultsFailed = dfResultsFailed.set_index([('', 'Op Name'), ('', 'Node Name')], drop=True)
                dfResultsFailed.index.names = ['Op Name', 'Node Name']
                dfResultsFailed.sort_index(inplace=True)

                htmlOutputDir = os.path.join(self.outputDir, 'html')
                makeSubdir(htmlOutputDir)
                inputFilename = os.path.basename(inputFile)
                with open(os.path.join(htmlOutputDir, quantizationVariation + '_' + inputFilename + '.html'), 'w') as f:
                    f.write(self.htmlHeadBegin.format(title=quantizationVariation + ' - ' + inputFilename) + self.htmlHeadEnd + self.htmlBody.format(legend=self.__getLegendTable(readableAlgorithmNames, algoDescriptions), instructions='Please consult the latest logs for further details on the failures.', summaryTitle='Summary of failed nodes that should be inspected: (Total number of nodes analyzed: ' + str(len(noAlgorithms)) + ' Total number of failed nodes: ' + str(len(noAlgorithmsFailed)) + ')', title='Results for quantizer: ' + quantizationVariation + ' using input file: ' + inputFilename + ' on model: ' + os.path.basename(self.inputModel), allNodesTitle='Results for all nodes:', summary=dfResultsFailed.to_html(justify='center', index=True, escape=False), table=dfResults.to_html(justify='center', index=True, escape=False)))
        self.__printInputDataToHtml(results, htmlOutputDir)

    def __printInputDataToHtml(self, results, htmlOutputDir):
        algorithmNames = self.__getInputAnalysisAlgoNames()
        readableAlgorithmNames = self.__translateAlgoNames(algorithmNames)
        readableAlgorithmNames.append('Description')
        columns = self.__getMultiIndexForColumns(readableAlgorithmNames)
        algoDescriptions = self.__generateAlgorithmDescriptions(readableAlgorithmNames)
        noAlgorithms = self.__getResultWithoutAlgos(results["inputData"])
        filenames = [row[0] for row in noAlgorithms]
        onlyAlgorithms = self.__getResultOnlyAlgos(results["inputData"], algorithmNames)
        dfOnlyAlgos = pd.DataFrame(onlyAlgorithms, columns=columns)
        dfNoAlgos = pd.DataFrame(filenames, columns=pd.MultiIndex.from_product([list(['']), list(['File Name'])]))
        dfResults = pd.concat([dfNoAlgos, dfOnlyAlgos], axis=1)
        dfResults = dfResults.set_index([('', 'File Name')], drop=True)
        dfResults.index.names = ['File Name']
        with open(os.path.join(htmlOutputDir, 'input_data_analysis.html'), 'w') as f:
            f.write(self.htmlHeadBegin.format(title='Input Data Analysis') + self.htmlHeadEnd + self.htmlBody.format(legend=self.__getLegendTable(readableAlgorithmNames, algoDescriptions), instructions='Please consult the latest csv or log files for further details on the analysis.', summaryTitle='Total number of input files analyzed: ' + str(len(filenames)), title='Results for input data analysis for the model: ' + os.path.basename(self.inputModel), allNodesTitle='Analysis for all input files:', summary='', table=dfResults.to_html(justify='center', index=True, escape=False)))

    def __getFailedNodes(self, results) -> List:
        failedNodes = []
        for result in results:
            # Exclude failed cases for Clustering of Unquantized Data from the summary, used only as additional information for now.
            if Verifiers.DATA_DISTRIBUTION in result.getResults():
                if self.__FAILDIV in list(result.getResults().values())[:-1]:
                    failedNodes.append(result)
            elif self.__FAILDIV in result.getResults().values():
                failedNodes.append(result)
        return failedNodes

    def __formatWeightsForLog(self) -> Dict:
        results = {}
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.weightResults:
                continue
            quantOptionResults = []
            for opAndNodeName, comparisonResults in self.weightResults[quantizationVariation].items():
                for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                    if comparisonResult != dict():
                        # ['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used']
                        # [opName, tensorName, perAlgorithmResult[1][0], perAlgorithmResult[1][1], perAlgorithmResult[1][2]]
                        quantOptionResults.append([opAndNodeName[0], opAndNodeName[1], comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName])
            quantOptionResults.sort(key=lambda x: x[2])
            results[quantizationVariation] = quantOptionResults
        return results

    def __formatBiasesForLog(self):
        results = {}
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.biasResults:
                continue
            quantOptionResults = []
            for opAndNodeName, comparisonResults in self.biasResults[quantizationVariation].items():
                for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                    if comparisonResult != dict():
                        # ['Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used']
                        # [opName, tensorName, perAlgorithmResult[1][0], perAlgorithmResult[1][1], perAlgorithmResult[1][2]]
                        quantOptionResults.append([opAndNodeName[0], opAndNodeName[1], comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName])
            quantOptionResults.sort(key=lambda x: x[2])
            results[quantizationVariation] = quantOptionResults
        return results

    def __formatActivationsForLog(self):
        results = {}
        for quantizationVariation in self.quantizationVariations[1:]:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in self.activationResults:
                continue
            quantOptionResults = []
            inputFileResults = self.activationResults[quantizationVariation]
            for activationNodeName, inputFileResult in inputFileResults.items():
                for inputFilename, comparisonResults in inputFileResult.items():
                    for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                        if comparisonResult != dict():
                            # ['Op Name', nodeType, 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename']
                            # [opName, tensorName, perAlgorithmResult[1][0], perAlgorithmResult[1][1], perAlgorithmResult[1][2], perAlgorithmResult[0]]
                            quantOptionResults.append([activationNodeName[0], activationNodeName[1], comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName, inputFilename])
            quantOptionResults.sort(key=lambda x: x[2])
            results[quantizationVariation] = quantOptionResults
        return results

    def __printTableToLog(self, header, results):
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in results:
                continue
            self.logger.print("", PrintOptions.LOGFILE)
            self.logger.print("Results for the " + quantizationVariation + " quantization:", PrintOptions.LOGFILE)
            results[quantizationVariation].insert(0, header)
            table = Table(results[quantizationVariation], True)
            logNoResults = True
            for row in table.getRows():
                cells = row.getCells()
                if cells[2].getString() not in (None, ''):
                    self.logger.print(table.decorate(row), PrintOptions.LOGFILE)
                    if not table.isFirstRow():
                        logNoResults = False
            if logNoResults:
                row = [Cell("N/A") for item in header]
                self.logger.print(table.decorate(Row(row)), PrintOptions.LOGFILE)

    def printLog(self):
        if self.activationResults is not None:
            self.logger.print(NEWLINE + NEWLINE + '<====ACTIVATIONS ANALYSIS====>' + NEWLINE, PrintOptions.LOGFILE)
            activationsForLog = self.__formatActivationsForLog()
            self.__printTableToLog(['Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename'], activationsForLog)

        self.logger.print(NEWLINE + NEWLINE + '<====WEIGHTS ANALYSIS====>' + NEWLINE, PrintOptions.LOGFILE)
        weightsForLog = self.__formatWeightsForLog()
        self.__printTableToLog(['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used'], weightsForLog)
        self.logger.print(NEWLINE + NEWLINE + '<====BIASES ANALYSIS====>' + NEWLINE, PrintOptions.LOGFILE)
        biasesForLog = self.__formatBiasesForLog()
        self.__printTableToLog(['Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used'], biasesForLog)

    def __formatWeightsForConsole(self):
        results = {}
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation not in self.weightResults:
                continue
            quantOptionResults = []
            for opAndNodeName, comparisonResults in self.weightResults[quantizationVariation].items():
                for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                    if comparisonResult != dict():
                        # ['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used']
                        # [opName, tensorName, perAlgorithmResult[1][0], perAlgorithmResult[1][1], perAlgorithmResult[1][2]]
                        if comparisonResult['pass'] == False:
                            quantOptionResults.append([opAndNodeName[0], opAndNodeName[1], comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName])
            results[quantizationVariation] = quantOptionResults
        return results

    def __formatBiasesForConsole(self):
        results = {}
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation not in self.biasResults:
                continue
            quantOptionResults = []
            for opAndNodeName, comparisonResults in self.biasResults[quantizationVariation].items():
                for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                    if comparisonResult != dict():
                        # ['Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used']
                        # [opName, tensorName, perAlgorithmResult[1][0], perAlgorithmResult[1][1], perAlgorithmResult[1][2]]
                        if comparisonResult['pass'] == False:
                            quantOptionResults.append([opAndNodeName[0], opAndNodeName[1], comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName])
            results[quantizationVariation] = quantOptionResults
        return results

    def __formatActivationsForConsole(self):
        results = {}
        for quantizationVariation in self.quantizationVariations[1:]:
            if quantizationVariation not in self.activationResults:
                continue
            quantOptionResults = []
            inputFileResults = self.activationResults[quantizationVariation]
            for activationNodeName, inputFileResult in inputFileResults.items():
                for inputFilename, comparisonResults in inputFileResult.items():
                    for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
                        # ['Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename']
                        # [opName, tensorName, perAlgorithmResult[1][0], perAlgorithmResult[1][1], perAlgorithmResult[1][2], perAlgorithmResult[0]]
                        if comparisonResult['pass'] == False:
                            quantOptionResults.append([activationNodeName[0], activationNodeName[1], comparisonResult['pass'], comparisonResult['data'], comparisonResult['threshold'], comparisonAlgorithmName, inputFilename])
            results[quantizationVariation] = quantOptionResults
        return results

    def __printTableToConsole(self, header, results):
        for quantizationVariation in self.quantizationVariations:
            # skip quantized models which are failed to get converted correctly
            if quantizationVariation not in results:
                continue
            self.logger.print("", PrintOptions.CONSOLE)
            self.logger.print("Results for the " + quantizationVariation + " quantization:", PrintOptions.CONSOLE)
            results[quantizationVariation].insert(0, header)
            table = Table(results[quantizationVariation], True)
            logNoResults = True
            for row in table.getRows():
                cells = row.getCells()
                if cells[2].getString() not in (None, ''):
                    self.logger.print(table.decorate(row), PrintOptions.CONSOLE)
                    if not table.isFirstRow():
                        logNoResults = False
            if logNoResults:
                row = [Cell("N/A") for item in header]
                self.logger.print(table.decorate(Row(row)), PrintOptions.CONSOLE)

    def printConsole(self):
        if self.activationResults is not None:
            self.logger.print(NEWLINE + NEWLINE + '<====ACTIVATIONS ANALYSIS FAILURES====>' + NEWLINE, PrintOptions.CONSOLE)
            activationsForConsole = self.__formatActivationsForConsole()
            self.__printTableToConsole(['Op Name', 'Activation Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used', 'Input Filename'], activationsForConsole)

        self.logger.print(NEWLINE + NEWLINE + '<====WEIGHTS ANALYSIS FAILURES====>' + NEWLINE, PrintOptions.CONSOLE)
        weightsForConsole = self.__formatWeightsForConsole()
        self.__printTableToConsole(['Op Name', 'Weight Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used'], weightsForConsole)
        self.logger.print(NEWLINE + NEWLINE + '<====BIASES ANALYSIS FAILURES====>' + NEWLINE, PrintOptions.CONSOLE)
        biasesForConsole = self.__formatBiasesForConsole()
        self.__printTableToConsole(['Op Name', 'Bias Node', 'Passes Accuracy', 'Accuracy Difference', 'Threshold Used', 'Algorithm Used'], biasesForConsole)

    def __formatActivationsForCsv(self, quantizationVariation) -> Tuple:
        quantOptionResults = {}
        quantOptionResultsHeader = {}
        inputFileResults = self.activationResults[quantizationVariation]
        for activationNodeName, inputFileResult in inputFileResults.items():
            for inputFilename, comparisonResults in inputFileResult.items():
                DataFormatter.__formatDataForCsv(activationNodeName[0], comparisonResults, quantOptionResults, quantOptionResultsHeader, activationNodeName[1], inputFilename)
        return (quantOptionResults, quantOptionResultsHeader)

    def __formatWeightsForCsv(self, quantizationVariation) -> Tuple:
        quantOptionResults = {}
        quantOptionResultsHeader = {}
        for opAndNodeName, comparisonResults in self.weightResults[quantizationVariation].items():
            DataFormatter.__formatDataForCsv(opAndNodeName[0], comparisonResults, quantOptionResults, quantOptionResultsHeader, opAndNodeName[1])
        return (quantOptionResults, quantOptionResultsHeader)

    def __formatBiasesForCsv(self, quantizationVariation) -> Tuple:
        quantOptionResults = {}
        quantOptionResultsHeader = {}
        for opAndNodeName, comparisonResults in self.biasResults[quantizationVariation].items():
            DataFormatter.__formatDataForCsv(opAndNodeName[0], comparisonResults, quantOptionResults, quantOptionResultsHeader, opAndNodeName[1])
        return (quantOptionResults, quantOptionResultsHeader)

    def __formatInputsForCsv(self) -> Tuple:
        inputAnalysisResults = {}
        inputAnalysisResultsHeader = {}
        for filename in self.inputResults:
            DataFormatter.__formatDataForCsv(filename, self.inputResults[filename], inputAnalysisResults, inputAnalysisResultsHeader)
        return (inputAnalysisResults, inputAnalysisResultsHeader)

    @staticmethod
    def __formatDataForCsv(opName, comparisonResults, quantOptionResults, quantOptionResultsHeader, nodeName=None, inputFilename=None):
        for comparisonAlgorithmName, comparisonResult in comparisonResults.items():
            if comparisonResult != dict():
                perComparisonAlgorithmResults = []
                if comparisonAlgorithmName in quantOptionResults.keys():
                    perComparisonAlgorithmResults = quantOptionResults[comparisonAlgorithmName]
                result = [opName, nodeName, comparisonResult['pass'], comparisonResult['threshold']]
                if inputFilename:
                    result.append(inputFilename)
                resultData = re.findall(r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?', str(comparisonResult['data']))
                resultHeader = re.findall(r'[a-zA-Z ]+[a-zA-Z]+[^ 0-9Ee]', str(comparisonResult['data']))
                result[3:3] = resultData
                perComparisonAlgorithmResults.append(result)
                quantOptionResults[comparisonAlgorithmName] = perComparisonAlgorithmResults
                quantOptionResultsHeader[comparisonAlgorithmName] = resultHeader

    def printCsv(self):
        csvOutputDir = os.path.join(self.outputDir, 'csv')
        makeSubdir(csvOutputDir)
        inputsForCsv = self.__formatInputsForCsv()
        DataFormatter.writeResultsToCsv(csvOutputDir, inputsForCsv, self.inputsWeightsAndBiasesAnalysisCsvHeader, 'input_data_analysis.csv')
        for quantizationVariation in self.quantizationVariations:
            if quantizationVariation != 'unquantized' and quantizationVariation in self.activationResults:
                activationsForCsv = self.__formatActivationsForCsv(quantizationVariation)
                DataFormatter.writeResultsToCsv(csvOutputDir, activationsForCsv, self.activationsAnalysisCsvHeader, quantizationVariation + '_activations.csv')
            if quantizationVariation in self.weightResults:
                weightsForCsv = self.__formatWeightsForCsv(quantizationVariation)
                DataFormatter.writeResultsToCsv(csvOutputDir, weightsForCsv, self.inputsWeightsAndBiasesAnalysisCsvHeader, quantizationVariation + '_weights.csv')
            if quantizationVariation in self.biasResults:
                biasesForCsv = self.__formatBiasesForCsv(quantizationVariation)
                DataFormatter.writeResultsToCsv(csvOutputDir, biasesForCsv, self.inputsWeightsAndBiasesAnalysisCsvHeader, quantizationVariation + '_biases.csv')

    @staticmethod
    def writeResultsToCsv(csvOutputDir, results, headers, filename):
        resultsForCsvItems = results[0].items()
        resultsForCsvAlgorithmHeaders = results[1]
        resultsCsvPath = os.path.join(csvOutputDir, filename)
        deleteFile(resultsCsvPath)
        with open(resultsCsvPath, 'a') as resultsCsv:
            for algorithmName, resultsData in resultsForCsvItems:
                np.savetxt(resultsCsv, ['Verifier Name: ' + algorithmName], delimiter=',', fmt='%s', comments='')
                np.savetxt(resultsCsv, resultsData, delimiter=',', fmt='%s', comments='', header=(', ').join(headers[:3] + resultsForCsvAlgorithmHeaders[algorithmName] + headers[3:]))
