#=============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

import qti.aisw.quantization_checker.utils.Verifiers as Verifiers
from typing import Dict, List

class ResultNode:
    def __init__(self, nodeName, tensorName, tensorType, results):
        self.__FAILDIV = '<div class=fail>Fail</div>'
        self.__nodeName = nodeName
        self.__tensorName = tensorName
        self.__tensorType = tensorType
        self.__results = results
        self.__failedVerifiers = self.__getListOfFailedVerifiers()
        self.__analysisDescription = Verifiers.getFailureAnalysisDescription(self.__failedVerifiers)

    def __getListOfFailedVerifiers(self) -> List:
        failedVerifiers = []
        if self.__FAILDIV in self.__results.values():
            for algoName, result in self.__results.items():
                if result == self.__FAILDIV:
                    failedVerifiers.append(algoName)
        return failedVerifiers

    def getNodeName(self) -> str:
        return self.__nodeName

    def getTensorName(self) -> str:
        return self.__tensorName

    def getTensorType(self) -> str:
        return self.__tensorType

    def getResults(self) -> Dict:
        return self.__results

    def getAnalysisDescription(self) -> str:
        return self.__analysisDescription


