cmake_minimum_required(VERSION 3.10) 

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 21) # API level
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)

if(NOT CMAKE_ANDROID_NDK)
    set(CMAKE_ANDROID_NDK ${CMAKE_SOURCE_DIR}/android-ndk-r19c)
endif()
# 设置android stl类型
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)
set(CMAKE_ANDROID_STL_TYPE c++_shared)
add_compile_definitions(ENABLE_GL_BUFFER)

set(PLATFORM "aarch64-android-clang" CACHE STRING "")

project(snpe-sample)

set(SNPE_LIB_DIR ${CMAKE_SOURCE_DIR}/3rdparty/snpe-2.12.0.230626/lib/aarch64-android)

set(SNPE_INCLUDE_DIR 
    ${CMAKE_SOURCE_DIR}/3rdparty/snpe-2.12.0.230626/include
    )

add_executable(snpe-sample
    src/LoadContainer.cpp
    src/NV21Load.cpp
    src/SaveOutputTensor.cpp
    src/main.cpp
    src/Util.cpp
    src/PreprocessInput.cpp
    src/LoadUDOPackage.cpp
    src/CreateGLContext.cpp
    src/SetBuilderOptions.cpp
    src/CreateGLBuffer.cpp
    src/LoadInputTensor.cpp
    src/CheckRuntime.cpp
    src/CreateUserBuffer.cpp
)

target_compile_options(snpe-sample PRIVATE -DENBALE_GL_BUFFER)

target_link_libraries(snpe-sample PRIVATE GLESv2 EGL log)

add_library(libSNPE SHARED IMPORTED)
message(STATUS "SNPE_LIB_DIR: ${SNPE_LIB_DIR}")
set_target_properties(libSNPE PROPERTIES IMPORTED_LOCATION ${SNPE_LIB_DIR}/libSNPE.so)

include_directories(
    ${SNPE_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
target_include_directories(snpe-sample PRIVATE $(SNPE_INCLUDE_DIR))
target_link_libraries(snpe-sample PRIVATE libSNPE)