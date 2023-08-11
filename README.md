#### 使用 SNPE SDK进行模型推理


## 1. 简介
- 本项目是基于 SNPE 进行模型推理的 DEMO.

## 2. 编译环境依赖
- Ubuntu LTS or Ubuntu docker
- CMake 3.13+

### 3. 编译流程
- 1. 下载 SNPE SDK [SNPE SDK 下载地址](https://qpm.qualcomm.com/#/main/tools/details/qualcomm_neural_processing_sdk)
- 2. 下载 NDK [NDK 下载地址](https://developer.android.google.cn/ndk/downloads/older_releases)
- 3. 编译 
    ```shell
        ./b.sh
    ```

### 4. Snapdragon 设备运行
    <!-- 将编译出来的可执行文件推到设备上 -->
    adb shell "mkdir -p /data/local/tmp/snpe_lab"
    adb push ./build/snpe-sample /data/local/tmp/snpe_lab
    <!-- 将依赖库推到设备上 -->
    adb shell "mkdir -p /data/local/tmp/snpe_lib"
    adb push ./3rdparty/snpe-2.12.0.230626/lib/aarch64-android/ /data/local/tmp/snpe_lib
    <!-- 运行 -->
    adb shell
    export SNPE_TARGET_ARCH=aarch64-android-clang8.0
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/snpe_lib/$SNPE_TARGET_ARCH
    cd /data/local/tmp/snpe_lab
    ./snpe-sample -h

### 5. 多 backend 支持
    ```
    DESCRIPTION:
    ------------
    Example application demonstrating how to load and execute a neural network
    using the SNPE C++ API.
    
    
    REQUIRED ARGUMENTS:
    -------------------
      -d  <FILE>   Path to the DL container containing the network.
      -i  <FILE>   Path to a file listing the inputs for the network.
      -o  <PATH>   Path to directory to store output results.
    
    OPTIONAL ARGUMENTS:
    -------------------
      -b  <TYPE>   Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16] (ITENSOR is default).
      -q  <BOOL>    Specifies to use static quantization parameters from the model instead of input specific quantization [true, false]. Used in conjunction with USERBUFFER_TF8. 
      -r  <RUNTIME> The runtime to be used [gpu, dsp, aip, cpu] (cpu is default). 
      -u  <VAL,VAL> Path to UDO package with registration library for UDOs. 
                    Optionally, user can provide multiple packages as a comma-separated list. 
      -z  <NUMBER>  The maximum number that resizable dimensions can grow into. 
                    Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer and is not applicable when using ITensor. 
      -c           Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.
      -l  <VAL,VAL,VAL> Specifies the order of precedence for runtime e.g  cpu_float32, dsp_fixed8_tf etc. Valid values are:- 
                        cpu_float32 (Snapdragon CPU)       = Data & Math: float 32bit 
                        gpu_float32_16_hybrid (Adreno GPU) = Data: float 16bit Math: float 32bit 
                        dsp_fixed8_tf (Hexagon DSP)        = Data & Math: 8bit fixed point Tensorflow style format 
                        gpu_float16 (Adreno GPU)           = Data: float 16bit Math: float 16bit 
                        cpu (Snapdragon CPU)               = Same as cpu_float32 
                        gpu (Adreno GPU)                   = Same as gpu_float32_16_hybrid 
                        dsp (Hexagon DSP)                  = Same as dsp_fixed8_tf 
    ```