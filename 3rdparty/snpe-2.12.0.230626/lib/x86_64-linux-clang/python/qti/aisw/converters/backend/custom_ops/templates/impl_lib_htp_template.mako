<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%operators = package.package_info.operators%>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================

#include "QnnHtpCommon.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include "optimize.h"
#include "op_register_ext.h"
#include "HTP/core/unique_types.h"
#include "HTP/core/simple_reg.h"

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()

/** Note that the order of declarations given here defines the order in which ops and graph optimizations are
 *  registered to the HTP Core.
 *  Append the latest OpName at the bottom
 */
% for operator in operators:
DECLARE_PKG_OPS_OPTS_LIST(PKG_${operator.type_name})
%endfor

END_PKG_OPS_OPTS_LIST()

static constexpr auto NUM_OPS = ${len(operators)};

// op package information
static constexpr auto packageName = "${package_name}";
static std::array<const char*, NUM_OPS> opNames{{""}};

static Qnn_ApiVersion_t sdkApiVersion            = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t implementationLibInfo = {packageName,
                                                    opNames.data(),
                                                    nullptr,
                                                    opNames.size(),
                                                    nullptr,
                                                    0,
                                                    QNN_SDK_BUILD_ID,
                                                    &sdkApiVersion,
                                                    nullptr,
                                                    0};

// global data
static QnnOpPackage_GlobalInfrastructure_t infra = nullptr;
static bool lib_initialized = false;

INIT_PACKAGE_OP_DEF()

INIT_PACKAGE_OPTIMIZATION_DEF()

Qnn_ErrorHandle_t SnpeUdo_initImplLibrary(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
  if (lib_initialized)
      return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;

  REGISTER_PACKAGE_OPS()
  REGISTER_PACKAGE_OPTIMIZATIONS()

  infra = infrastructure;
  lib_initialized = true;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t SnpeUdo_getImpInfo(const QnnOpPackage_Info_t** info) {
  if (!lib_initialized)
      return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

  *info = &implementationLibInfo;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t SnpeUdo_terminateImplLibrary() {
  if (!lib_initialized)
      return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

  infra = nullptr;
  lib_initialized = false;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t SnpeUdo_logInitialize(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t SnpeUdo_logSetLevel(QnnLog_Level_t maxLogLevel) {
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t SnpeUdo_logTerminate() {
  return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t SnpeUdo_ValidateOpConfig (Qnn_OpConfig_t opConfig){
return QNN_SUCCESS;
}

Qnn_ErrorHandle_t ${package_name}InterfaceProvider(QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion      = {1, 4, 0};
  interface->v1_4.init             = SnpeUdo_initImplLibrary;
  interface->v1_4.terminate        = SnpeUdo_terminateImplLibrary;
  interface->v1_4.getInfo          = SnpeUdo_getImpInfo;
  interface->v1_4.validateOpConfig = SnpeUdo_ValidateOpConfig;
  interface->v1_4.createOpImpl     = nullptr;
  interface->v1_4.freeOpImpl       = nullptr;
  interface->v1_4.logInitialize    = nullptr;
  interface->v1_4.logSetLevel      = SnpeUdo_logSetLevel;
  interface->v1_4.logTerminate     = SnpeUdo_logTerminate;
  return QNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
