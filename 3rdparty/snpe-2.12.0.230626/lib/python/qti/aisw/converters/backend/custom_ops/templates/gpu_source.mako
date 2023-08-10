<%doc>
//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================</%doc>
<%page expression_filter="n" expression_filter="trim" />
//==============================================================================
// Auto Generated Code for ${package_info.name}
//==============================================================================

#include "Operation.hpp"
#include "QnnGpuOpPackage.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"

const std::string ${operator.type_name}Operation::s_operationType = "${operator.type_name}";

std::shared_ptr<Operation> ${operator.type_name}Operation::create(const QnnGpuOpPackage_Node_t* node,
                                                 Qnn_ErrorHandle_t* status) {
  return std::shared_ptr<${operator.type_name}Operation>(new (std::nothrow) ${operator.type_name}Operation(node, status));
}

${operator.type_name}Operation::${operator.type_name}Operation(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status)
    : Operation() {

  /**
   * Add code here
   **/

  auto kernel = setKernelInfo(node, status);
  if (QNN_SUCCESS != *status) {
    return;
  }
  m_kernels.push_back(kernel);

  RETURN(QNN_SUCCESS);
}

QnnGpu_Kernel_t ${operator.type_name}Operation::setKernelInfo(const QnnGpuOpPackage_Node_t* node, Qnn_ErrorHandle_t* status) {
  QnnGpu_Kernel_t gpuKernel = QNN_GPU_KERNEL_INIT;

  /**
   * Add code here
   **/

  KERNEL_RETURN(QNN_SUCCESS);
}
