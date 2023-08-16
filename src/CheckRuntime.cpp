#include <iostream>

#include "CheckRuntime.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/String.hpp"

// Command line settings
zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime, bool &staticQuantization)
{
    // 获取当前 SNPE 的版本号, 并打印出来
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number

   // 检查 runtime 是否为 DSP, 以及是否开启了 静态量化 
   if((runtime != zdl::DlSystem::Runtime_t::DSP) && staticQuantization)
   {
      // 如果运行时环境不是DSP,但启用了静态量化,会打印错误,并将staticQuantization设置为false,因为静态量化只适用于DSP/AIP运行时环境
      std::cerr << "ERROR: Cannot use static quantization with CPU/GPU runtimes. It is only designed for DSP/AIP runtimes.\n";
      std::cerr << "ERROR: Proceeding without static quantization on selected runtime.\n";
      staticQuantization = false;
   }

    // 用 SNPEFactory的isRuntimeAvailable()函数检查所选运行时环境是否可用, 如果不能用就 Fall back 到 cpu_runtime
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
    {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    return runtime;
}

// 调用SNPEFactory的isGLCLInteropSupported()静态函数, 检查当前SNPE 运行环境是否支持 OpenCL 和 OpenGL 之间的互操作, (GLCL Interop)
bool checkGLCLInteropSupport()
{
    return zdl::SNPE::SNPEFactory::isGLCLInteropSupported();
}
