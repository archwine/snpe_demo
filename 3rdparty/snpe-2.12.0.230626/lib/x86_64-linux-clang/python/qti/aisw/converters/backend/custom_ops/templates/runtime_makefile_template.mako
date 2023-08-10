<%doc>
# ==============================================================================
#
#  Copyright (c) 2022 - 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
#This is a common template for all the runtimes.
#For DSP runtime, depending on the architecture this will generate makefile.
</%doc>
#================================================================================
# Auto Generated Code for ${package.name}
#================================================================================

# define relevant directories
SRC_DIR := ./

%if str(runtime).lower() != 'dsp':
# define library name and corresponding directory
%if str(runtime).lower() != 'cpu':
export RUNTIME := ${str(runtime).upper()}
export LIB_DIR := ../../../libs/$(TARGET)/$(RUNTIME)
%else:
export LIB_DIR := ../../../libs/$(TARGET)
%endif

library := $(LIB_DIR)/libUdo${package.name}Impl${runtime}.so

%if str(runtime).lower() == 'gpu':
# Note: add CL include path here to compile Gpu Library or set as env variable
# export CL_INCLUDE_PATH = <my_cl_include_path>
%endif

# define target architecture if not previously defined, default is x86
ifndef TARGET_AARCH_VARS
TARGET_AARCH_VARS:= -march=x86-64
endif

# specify package paths, should be able to override via command line?
UDO_PACKAGE_ROOT =${package.root}

include ../../../common.mk

%else:
# NOTE:
# - this Makefile is going to be used only to create DSP skels, so no need for android.min
<%DSP_ARCH = int((str(dsp_arch_type).lower())[-2:])%>

%if DSP_ARCH >= 68:
%if DSP_ARCH >= 73:
%if DSP_ARCH >= 75:
HEXAGON_SDK_VERSION = 5.3.0
HEXAGON_TOOLS_VERSION = 8.7.03
%else:
HEXAGON_SDK_VERSION = 5.0.0
HEXAGON_TOOLS_VERSION = 8.5.10
%endif
%else:
HEXAGON_SDK_VERSION = 4.2.0
HEXAGON_TOOLS_VERSION = 8.4.09
%endif
ifdef HEXAGON_SDK4_ROOT
HEXAGON_SDK_ROOT = $(HEXAGON_SDK4_ROOT)
HEXAGON_TOOLS_ROOT = $(HEXAGON_SDK4_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)
endif
%else:
HEXAGON_SDK_VERSION = 3.5.2
HEXAGON_TOOLS_VERSION = 8.3.07
%endif
ifndef HEXAGON_SDK_ROOT
$(error "HEXAGON_SDK_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_SDK_ROOT to hexagon sdk installation.(Supported Version :$(HEXAGON_SDK_VERSION))"
endif

ifndef HEXAGON_TOOLS_ROOT
$(error "HEXAGON_TOOLS_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_TOOLS_ROOT to HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)")
endif

%if DSP_ARCH >= 68:
ifndef QNN_SDK_ROOT
$(error "QNN_SDK_ROOT needs to be defined to compile a dsp library. Please set QNN_SDK_ROOT to qnn sdk installation.")
endif
%endif

ifndef SDK_SETUP_ENV
$(error "SDK_SETUP_ENV needs to be defined to compile a dsp library. Please set SDK_SETUP_ENV=Done")
endif

%if DSP_ARCH >= 68:
# define variant as V=hexagon_Release_dynamic_toolv84_v68 - it can be hardcoded too
ifndef V
V = hexagon_Release_dynamic_toolv84_v68
endif
%else:
# define variant as V=hexagon_Release_dynamic_toolv83_${dsp_arch_type} - it can be hardcoded too
ifndef V
V = hexagon_Release_dynamic_toolv83_${dsp_arch_type}
endif
%endif

V_TARGET = $(word 1,$(subst _, ,$(V)))
ifneq ($(V_TARGET),hexagon)
$(error Unsupported target '$(V_TARGET)' in variant '$(V)')
endif

# define package include paths and check API header path
# set package include paths, note package root will take precedence
# if includes are already present in package
UDO_PACKAGE_ROOT =${package.root}
PKG_NAME = ${package.name}

# must list all variants supported by this project
SUPPORTED_VS = $(default_VS)

%if DSP_ARCH >= 68:
QNN_INCLUDE = $(QNN_SDK_ROOT)/include/QNN
QNN_HTP_INCLUDE = $(QNN_INCLUDE)/HTP

include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min

CXX_FLAGS += -std=c++17 -fvisibility=default -stdlib=libc++ -fexceptions -MMD -DTHIS_PKG_NAME=$(PKG_NAME)
CXX_FLAGS += -I$(QNN_INCLUDE) -I$(QNN_HTP_INCLUDE) -I$(QNN_HTP_INCLUDE)/core
CXX_FLAGS += -I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev${DSP_ARCH}/include/qurt -I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev${DSP_ARCH}/include/posix
CXX_FLAGS += $(MHVX_DOUBLE_FLAG) -mhmx -DUSE_OS_QURT
CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"

BUILD_DLLS = libUdo${package.name}Impl${runtime}

# sources for the DSP implementation library in src directory
SRC_DIR = ./
libUdo${package.name}Impl${runtime}.CXX_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

%else:
# must list all the dependencies of this project
DEPENDENCIES = ATOMIC RPCMEM TEST_MAIN TEST_UTIL

# each dependency needs a directory definition
#  the form is <DEPENDENCY NAME>_DIR
#  for example:
#    DEPENDENCIES = FOO
#    FOO_DIR = $(HEXAGON_SDK_ROOT)/examples/common/foo
#

ATOMIC_DIR = $(HEXAGON_SDK_ROOT)/libs/common/atomic
RPCMEM_DIR = $(HEXAGON_SDK_ROOT)/libs/common/rpcmem
TEST_MAIN_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_main
TEST_UTIL_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_util

include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min

# set include paths as compiler flags
CC_FLAGS += -I $(UDO_PACKAGE_ROOT)/include

# if SNPE_ROOT is set and points to the SDK path, it will be used. Otherwise ZDL_ROOT will be assumed to point
# to a build directory
ifdef SNPE_ROOT
CC_FLAGS += -I $(SNPE_ROOT)/include/zdl
else ifdef ZDL_ROOT
CC_FLAGS += -I $(ZDL_ROOT)/x86_64-linux-clang/include/zdl
else
$(error SNPE_ROOT: Please set SNPE_ROOT or ZDL_ROOT to obtain Udo headers necessary to compile the package)
endif

ifndef QNN_INCLUDE
$(info "INFO: Qnn include not explicitly defined, attempting to use QNN_SDK_ROOT if it is valid")
QNN_INCLUDE := $(QNN_SDK_ROOT)/include/QNN
endif
ifeq ($(wildcard $(QNN_INCLUDE)),)
$(error "ERROR: QNN_INCLUDE path is not set. QNN include paths must be set to obtain BE headers necessary to compile the package")
endif

ifndef PACKAGE_NAME
PACKAGE_NAME := $(notdir $(shell pwd))
$(info "INFO: No package name defined. Using current directory name: $(PACKAGE_NAME) as the package name")
endif

# set include paths as compiler flags
CC_FLAGS += -I $(SRC_DIR)/include
CC_FLAGS += -I $(QNN_INCLUDE)/

# only build the shared object if dynamic option specified in the variant
ifeq (1,$(V_dynamic))
BUILD_DLLS = libUdo${package.name}Impl${runtime}
endif

OP_SOURCES = $(wildcard $(SRC_DIR)/src/ops/*.cpp)
OTHER_SOURCES = $(wildcard $(SRC_DIR)/src/*.cpp)

# sources for the DSP implementation library in src directory
libUdo${package.name}Impl${runtime}.C_SRCS := $(wildcard $(OP_SOURCES)) $(OTHER_SOURCES)

%endif

# copy final build products to the ship directory
BUILD_COPIES = $(DLLS) $(EXES) $(LIBS) $(UDO_PACKAGE_ROOT)/libs/dsp_${str(dsp_arch_type)}/

# always last
include $(RULES_MIN)

# define destination library directory, and copy files into lib/dsp
# this code will create it
SHIP_LIBS_DIR   := $(CURDIR)/$(V)
LIB_DIR         := $(UDO_PACKAGE_ROOT)/libs/dsp_${str(dsp_arch_type)}
OBJ_DIR         := $(UDO_PACKAGE_ROOT)/obj/local/dsp_${str(dsp_arch_type)}

.PHONY: dsp

dsp: tree
	mkdir -p ${"${OBJ_DIR}"};  ${"\\"}
	cp -Rf ${"${SHIP_LIBS_DIR}"}/. ${"${OBJ_DIR}"} ;${"\\"}
	rm -rf ${"${SHIP_LIBS_DIR}"};


%if DSP_ARCH >= 68:
X86_LIBNATIVE_RELEASE_DIR = $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/$(HEXAGON_TOOLS_VERSION)/Tools
X86_OBJ_DIR = $(UDO_PACKAGE_ROOT)/obj/local/x86-64_linux_clang/
X86_LIB_DIR = $(UDO_PACKAGE_ROOT)/libs/x86-64_linux_clang/

X86_CXX ?= clang++
ifeq ($(shell $(X86_CXX) -v 2>&1 | grep -c "clang version"), 0)
  X86_CXX := clang++-9
endif

X86_C__FLAGS = -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -DUSE_OS_LINUX
X86_CXXFLAGS = -std=c++17 -I$(QNN_INCLUDE) -I$(QNN_HTP_INCLUDE) -I$(QNN_HTP_INCLUDE)/core -fPIC -Wall -Wreorder -Wno-missing-braces -Werror -Wno-format -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++
X86_CXXFLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"
X86_CXXFLAGS += $(X86_C__FLAGS) -fomit-frame-pointer -Wno-invalid-offsetof -DTHIS_PKG_NAME=$(PKG_NAME)
X86_LDFLAGS =  -Wl,--whole-archive -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative -Wl,--no-whole-archive -lpthread

OBJS = $(patsubst %.cpp,%.o,$($(BUILD_DLLS).CXX_SRCS))

X86_DIR:
	mkdir -p $(X86_OBJ_DIR)
	mkdir -p $(X86_LIB_DIR)

$(X86_OBJ_DIR)/%.o: %.cpp
	$(X86_CXX) $(X86_CXXFLAGS) -MMD -c $< -o $@

$(X86_OBJ_DIR)/$(BUILD_DLLS).so: $(patsubst %,$(X86_OBJ_DIR)/%,$(OBJS))
	$(X86_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(X86_LDFLAGS)

X86_DLL: $(X86_OBJ_DIR)/$(BUILD_DLLS).so

dsp_x86:  X86_DIR X86_DLL
	mv $(X86_OBJ_DIR)/$(BUILD_DLLS).so $(X86_LIB_DIR)

# Setup compiler, compiler instructions and linker for aarch64

AARCH64_OBJ_DIR = $(UDO_PACKAGE_ROOT)/obj/local/arm64-v8a/
AARCH64_LIB_DIR = $(UDO_PACKAGE_ROOT)/libs/arm64-v8a/
QNN_CPU_INCLUDE = $(QNN_INCLUDE)/CPU
QNN_GPU_INCLUDE = $(QNN_INCLUDE)/GPU
AARCH64_C__FLAGS = -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include -ffast-math -DUSE_OS_LINUX -DANDROID
AARCH64_CXX_FLAGS = $(AARCH64_C__FLAGS) -I$(QNN_INCLUDE) -I$(QNN_HTP_INCLUDE) -I$(QNN_HTP_INCLUDE)/core -I$(QNN_CPU_INCLUDE) -I$(QNN_GPU_INCLUDE) -fomit-frame-pointer -Wno-invalid-offsetof  -Wno-unused-variable -Wno-unused-parameter -Wno-missing-braces -Wno-sign-compare -Wno-unused-private-field -Wno-unused-variable -Wno-ignored-qualifiers -Wno-missing-field-initializers
AARCH64_CXX_FLAGS += -std=c++17 -fPIC -Wall -Wreorder -Wno-missing-braces -Werror -Wno-format -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++ -DTHIS_PKG_NAME=$(PKG_NAME)
AARCH64_CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"  -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"
ARM_CLANG_OPTS =--target=aarch64-none-linux-android21 --sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot
AARCH64_CXX = $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ $(ARM_CLANG_OPTS)
AARCH64_LDFLAGS = -L$(SNPE_ROOT)/lib/aarch64-android -lSnpeHtpPrepare
AARCH64_DLL = libUdo${package.name}Impl${runtime}_AltPrep

libUdo${package.name}Impl${runtime}_AltPrep.CXX_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

AARCH64_OBJS = $(patsubst %.cpp,%.o,$($(AARCH64_DLL).CXX_SRCS))

AARCH64_DIR:
	mkdir -p $(AARCH64_OBJ_DIR)
	mkdir -p $(AARCH64_LIB_DIR)

$(AARCH64_OBJ_DIR)/%.o: %.cpp
	$(AARCH64_CXX) $(AARCH64_CXX_FLAGS) -MMD -c $< -o $@

$(AARCH64_OBJ_DIR)/$(AARCH64_DLL).so: $(patsubst %,$(AARCH64_OBJ_DIR)/%,$(AARCH64_OBJS))
	$(AARCH64_CXX) -fPIC -std=c++17 -g -shared -o $@ $^ $(AARCH64_LDFLAGS)

AARCH64_BUILD: $(AARCH64_OBJ_DIR)/$(AARCH64_DLL).so

dsp_aarch64:  AARCH64_DIR AARCH64_BUILD
	mv $(AARCH64_OBJ_DIR)/$(AARCH64_DLL).so $(AARCH64_LIB_DIR)

%endif
%endif
