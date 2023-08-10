//=============================================================================
//
//  Copyright (c) 2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================
#pragma once

#include <string>

#include "Wrapper.hpp"

#include "DlSystem/DlEnums.hpp"


#include "PlatformValidator/PlatformValidator.h"


namespace SNPE {

class PlatformValidator : public Wrapper<PlatformValidator, Snpe_PlatformValidator_Handle_t> {
  friend BaseType;
  using BaseType::BaseType;

  static constexpr DeleteFunctionType DeleteFunction{Snpe_PlatformValidator_Delete};

public:
  PlatformValidator()
    : BaseType(Snpe_PlatformValidator_Create())
  {  }

  void setRuntime(DlSystem::Runtime_t runtime){
    Snpe_PlatformValidator_SetRuntime(handle(), static_cast<Snpe_Runtime_t>(runtime));
  }

  bool isRuntimeAvailable(){
    return Snpe_PlatformValidator_IsRuntimeAvailable(handle());
  }

  std::string getCoreVersion(){
    return Snpe_PlatformValidator_GetCoreVersion(handle());
  }

  std::string getLibVersion(){
    return Snpe_PlatformValidator_GetLibVersion(handle());
  }

  bool runtimeCheck(){
    return Snpe_PlatformValidator_RuntimeCheck(handle());
  }

};

} // ns SNPE

ALIAS_IN_ZDL_NAMESPACE(SNPE, PlatformValidator)
