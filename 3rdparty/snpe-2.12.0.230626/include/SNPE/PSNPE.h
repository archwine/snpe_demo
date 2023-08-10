//==============================================================================
//
//  Copyright (c) 2022,2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _SNPE_PSNPE_H_
#define _SNPE_PSNPE_H_


#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#include "DlContainer/DlContainer.h"
#include "SNPE/ApplicationBufferMap.h"
#include "SNPE/RuntimeConfigList.h"
#include "SNPE/UserBufferList.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/IBufferAttributes.h"

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/DlError.h"

#include "DlSystem/UserMemoryMap.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef void* Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t;

//SNPE_API
//Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t Snpe_PSNPE_OutputAsyncCallbackParam_Create(size_t index,
//                                                                                        int status,
//                                                                                        const char* errorMsg);
//
//SNPE_API
//Snpe_ErrorCode_t Snpe_PSNPE_OutputAsyncCallbackParam_Delete(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

// NOTE: we don't need _{Create,Delete} functions because the user does not create or delete these handles
// They're passed in to the callback functions they created
SNPE_API
size_t Snpe_PSNPE_OutputAsyncCallbackParam_GetDataIdx(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

SNPE_API
int Snpe_PSNPE_OutputAsyncCallbackParam_GetExecuteStatus(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

SNPE_API
const char* Snpe_PSNPE_OutputAsyncCallbackParam_GetErrorMsg(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);

SNPE_API
size_t Snpe_PSNPE_OutputAsyncCallbackParam_GetID(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t oacpHandle);




typedef void* Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t;

SNPE_API
size_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetDataIdx(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

SNPE_API
int Snpe_PSNPE_InputOutputAsyncCallbackParam_GetExecuteStatus(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

SNPE_API
const char* Snpe_PSNPE_InputOutputAsyncCallbackParam_GetErrorMsg(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetUserBufferNames(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

SNPE_API
Snpe_ApplicationBufferMap_Handle_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetOutputMap_Ref(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);

SNPE_API
size_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetID(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle);


typedef void* Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t;

SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetInputs(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t ioiacpHandle);

SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetInputNames(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t ioiacpHandle);

SNPE_API
size_t Snpe_PSNPE_InputOutputInputAsyncCallbackParam_GetID(Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t ioiacpHandle);


// TODO: move to better location?
typedef struct{
  const uint8_t* data;
  size_t size;
} Snpe_UserBufferData_t;

SNPE_API
Snpe_UserBufferData_t Snpe_PSNPE_InputOutputAsyncCallbackParam_GetUserBuffer(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t ioacpHandle,
                                                                             const char* name);


typedef void* Snpe_BuildConfig_Handle_t;


typedef void* Snpe_PSNPE_Handle_t;



// TODO: nomenclature for these two enums? Also, this should be generated from PSNPE.hpp
typedef enum SNPE_API {
  SNPE_PSNPE_BUILDMODE_SERIAL = 0,
  SNPE_PSNPE_BUILDMODE_PARALLEL = 1
} Snpe_PSNPE_BuildMode_t;

typedef enum SNPE_API {
  SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_SYNC = 0,
  SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_OUTPUTASYNC = 1,
  SNPE_PSNPE_INPUTOUTPUTTRANSMISSIONMODE_INPUTOUTPUTASYNC = 2
} Snpe_PSNPE_InputOutputTransmissionMode_t;




// BuildConfig

SNPE_API
Snpe_BuildConfig_Handle_t Snpe_BuildConfig_Create();

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_Delete(Snpe_BuildConfig_Handle_t buildConfigHandle);

SNPE_API
Snpe_PSNPE_BuildMode_t Snpe_BuildConfig_GetBuildMode(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetBuildMode(Snpe_BuildConfig_Handle_t bcHandle, Snpe_PSNPE_BuildMode_t buildMode);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetContainer(Snpe_BuildConfig_Handle_t bcHandle, Snpe_DlContainer_Handle_t dlcHandle);

SNPE_API
Snpe_DlContainer_Handle_t Snpe_BuildConfig_GetContainer_Ref(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputBufferNames(Snpe_BuildConfig_Handle_t bcHandle, Snpe_StringList_Handle_t slHandle);

SNPE_API
Snpe_StringList_Handle_t Snpe_BuildConfig_GetOutputBufferNames_Ref(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputTensors(Snpe_BuildConfig_Handle_t bcHandle, Snpe_StringList_Handle_t slHandle);

SNPE_API
Snpe_StringList_Handle_t Snpe_BuildConfig_GetOutputTensors_Ref(Snpe_BuildConfig_Handle_t bcHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetRuntimeConfigList(Snpe_BuildConfig_Handle_t bcHandle, Snpe_RuntimeConfigList_Handle_t rclHandle);

SNPE_API
Snpe_RuntimeConfigList_Handle_t Snpe_BuildConfig_GetRuntimeConfigList_Ref(Snpe_BuildConfig_Handle_t bcHandle);

// TODO: why does PSNPE have them named "*ThreadNumbers" not "*ThreadNumber"?
SNPE_API
size_t Snpe_BuildConfig_GetInputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle, size_t threadNumbers);

SNPE_API
size_t Snpe_BuildConfig_GetOutputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputThreadNumbers(Snpe_BuildConfig_Handle_t bcHandle, size_t threadNumbers);



SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputCallback(Snpe_BuildConfig_Handle_t bcHandle,
                                                    void (*callbackFunc)(Snpe_PSNPE_OutputAsyncCallbackParam_Handle_t));


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetOutputCallbackID(Snpe_BuildConfig_Handle_t bcHandle, size_t id);


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_ClearOutputCallback(Snpe_BuildConfig_Handle_t bcHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputCallback(Snpe_BuildConfig_Handle_t bcHandle,
                                                    void (*callbackFunc)(Snpe_PSNPE_InputOutputAsyncCallbackParam_Handle_t));

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputCallbackID(Snpe_BuildConfig_Handle_t bcHandle, size_t id);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_ClearInputOutputCallback(Snpe_BuildConfig_Handle_t bcHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputInputCallback(Snpe_BuildConfig_Handle_t bcHandle,
                                                              Snpe_ApplicationBufferMap_Handle_t (*callbackFunc)(
                                                                Snpe_PSNPE_InputOutputInputAsyncCallbackParam_Handle_t
                                                              )
                                                              );
SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputInputCallbackID(Snpe_BuildConfig_Handle_t bcHandle, size_t id);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_ClearInputOutputInputCallback(Snpe_BuildConfig_Handle_t bcHandle);




SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetInputOutputTransmissionMode(Snpe_BuildConfig_Handle_t bcHandle,
                                                                 Snpe_PSNPE_InputOutputTransmissionMode_t iotMode);

SNPE_API
Snpe_PSNPE_InputOutputTransmissionMode_t Snpe_BuildConfig_GetInputOutputTransmissionMode(Snpe_BuildConfig_Handle_t bcHandle);



SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetProfilingLevel(Snpe_BuildConfig_Handle_t bcHandle, Snpe_ProfilingLevel_t profilingLevel);

SNPE_API
Snpe_ProfilingLevel_t Snpe_BuildConfig_GetProfilingLevel(Snpe_BuildConfig_Handle_t bcHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEncode(Snpe_BuildConfig_Handle_t bcHandle, uint64_t encode0, uint64_t encode1);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEncode0(Snpe_BuildConfig_Handle_t bcHandle, uint64_t encode0);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEncode1(Snpe_BuildConfig_Handle_t bcHandle, uint64_t encode1);

SNPE_API
uint64_t* Snpe_BuildConfig_GetEncode(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
uint64_t Snpe_BuildConfig_GetEncode0(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
uint64_t Snpe_BuildConfig_GetEncode1(Snpe_BuildConfig_Handle_t bcHandle);



SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetEnableInitCache(Snpe_BuildConfig_Handle_t bcHandle, int enableInitCache);

SNPE_API
int Snpe_BuildConfig_GetEnableInitCache(Snpe_BuildConfig_Handle_t bcHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetPlatformOptions(Snpe_BuildConfig_Handle_t bcHandle, const char* platformOptions);

SNPE_API
const char* Snpe_BuildConfig_GetPlatformOptions(Snpe_BuildConfig_Handle_t bcHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_BuildConfig_SetDiaglogOutputDir(Snpe_BuildConfig_Handle_t bcHandle, const char* diaglogOutputDir);

SNPE_API
const char* Snpe_BuildConfig_GetDiaglogOutputDir(Snpe_BuildConfig_Handle_t bcHandle);






SNPE_API
Snpe_PSNPE_Handle_t Snpe_PSNPE_Create();

SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_Delete(Snpe_PSNPE_Handle_t psnpeHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_Build(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_BuildConfig_Handle_t bcHandle);


SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_Execute(Snpe_PSNPE_Handle_t psnpeHandle,
                                    Snpe_UserBufferList_Handle_t inputBufferListHandle,
                                    Snpe_UserBufferList_Handle_t outputBufferListHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_ExecuteInputOutputAsync(Snpe_PSNPE_Handle_t psnpeHandle,
                                                    Snpe_StringList_Handle_t inputMapHandle,
                                                    size_t dataIndex,
                                                    int isTF8buff,
                                                    int isTF8Outputbuff);

// TODO: rmme, this was used for debugging
SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_ExecuteInputOutputAsyncTEST(Snpe_PSNPE_Handle_t psnpeHandle,
                                                    const void* stringVec,
                                                    size_t dataIndex,
                                                    int isTF8buff);

//SNPE_API
//Snpe_ErrorCode_t Snpe_PSNPE_ExecuteInputOutputAsync(Snpe_PSNPE_Handle_t psnpeHandle,
//                                                    Snpe_StringList_Handle_t inputMapHandle,
//                                                    size_t dataIndex,
//                                                    int isTF8buff,
//                                                    int isTF8Outputbuff);
SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_GetInputTensorNames(Snpe_PSNPE_Handle_t psnpeHandle);

SNPE_API
Snpe_StringList_Handle_t Snpe_PSNPE_GetOutputTensorNames(Snpe_PSNPE_Handle_t psnpeHandle);


SNPE_API
Snpe_TensorShape_Handle_t Snpe_PSNPE_GetInputDimensions(Snpe_PSNPE_Handle_t psnpeHandle);

SNPE_API
Snpe_TensorShape_Handle_t Snpe_PSNPE_GetInputDimensions_Name(Snpe_PSNPE_Handle_t psnpeHandle, const char* name);


SNPE_API
Snpe_TensorShape_Handle_t Snpe_PSNPE_GetBufferAttributesDims(Snpe_PSNPE_Handle_t psnpeHandle, const char* name);

SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_RegisterIonBuffers(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_UserMemoryMap_Handle_t ionBufferMapHandle);

SNPE_API
Snpe_ErrorCode_t Snpe_PSNPE_DeregisterIonBuffers(Snpe_PSNPE_Handle_t psnpeHandle, Snpe_StringList_Handle_t ionBufferNames);


SNPE_API
const char* Snpe_PSNPE_GetLastErrorString(Snpe_PSNPE_Handle_t psnpeHandle);

SNPE_API
Snpe_IBufferAttributes_Handle_t Snpe_PSNPE_GetInputOutputBufferAttributes(Snpe_PSNPE_Handle_t psnpeHandle, const char *name);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _SNPE_PSNPE_H_
