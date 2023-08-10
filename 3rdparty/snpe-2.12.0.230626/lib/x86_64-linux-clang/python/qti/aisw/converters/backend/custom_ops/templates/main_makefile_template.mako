<%doc>
# ==============================================================================
#
#  Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
#================================================================================
# Auto Generated Code for ${package.name}
#================================================================================

# define default
default: all

# define package name
PACKAGE_NAME := $(notdir $(shell pwd))

# define library prerequisites list
lib_cpu := jni/src/CPU
lib_reg := jni/src/reg
lib_gpu := jni/src/GPU
% if dsp_arch_types:
% for arch in dsp_arch_types:
lib_dsp_${arch} := jni/src/DSP_${str(arch).upper()}
% endfor
% else:
lib_dsp := jni/src/DSP
% endif

LIB_SOURCES = $(lib_cpu) $(lib_reg)\
% if dsp_arch_types:
% for arch in dsp_arch_types:
 $(lib_dsp_${arch})\
% endfor
% else:
 $(lib_dsp)
% endif

# define target_architecture
export TARGET_AARCH_VARS:= -march=x86-64

# define target name
export TARGET = x86-64_linux_clang

# specify compiler
export CXX := clang++
ifeq ($(shell $(CXX) -v 2>&1 | grep -c "clang version"), 0)
  export CXX := clang++-9
endif

# define default Android ABI
PLATFORM ?= arm64-v8a

.PHONY: all $(LIB_SOURCES) all_android all_x86 cpu dsp reg cpu_x86 dsp_android reg_x86 cpu_android gpu_android reg_android
all: $(LIB_SOURCES) all_x86 all_android

# Combined Targets
cpu: cpu_x86 cpu_android
gpu: gpu_android
dsp: dsp_android
reg: reg_x86 reg_android
clean: clean_x86 clean_android

# x86 Targets
all_x86: cpu_x86 reg_x86

cpu_x86: reg_x86
	$(call build_if_exists,$(lib_cpu),-$(MAKE) cpu_x86 -C $(lib_cpu))
	@cp $(lib_cpu)/libs/x86_64-linux-clang/libCPU.so libs/$(TARGET)/libUdo$(PACKAGE_NAME)ImplCpu.so

reg_x86:
	-$(MAKE) -C $(lib_reg)

% if dsp_arch_types:
% for arch in dsp_arch_types:
% if int((str(arch).lower())[-2:]) >= 68:
dsp_x86:
	$(call build_if_exists,$(lib_dsp_${arch}),$(MAKE) -C $(lib_dsp_${arch}) dsp_x86)
dsp_aarch64:
	$(call build_if_exists,$(lib_dsp_${arch}),$(MAKE) -C $(lib_dsp_${arch}) dsp_aarch64)
% endif
% endfor
% endif

clean_x86:
	$(call build_if_exists,$(lib_cpu),-$(MAKE) clean_x86 -C $(lib_cpu))
	@rm -rf libs obj

# Android Targets
NDK_CPU_IMPL_LIB := Udo$(PACKAGE_NAME)ImplCpu
NDK_GPU_IMPL_LIB := Udo$(PACKAGE_NAME)ImplGpu
NDK_REG_LIB := Udo$(PACKAGE_NAME)Reg

all_android: dsp_android warn_gpu check_ndk gpu_android cpu_android

dsp_android: reg_android
% if dsp_arch_types:
% for arch in dsp_arch_types:
	$(call build_if_exists,$(lib_dsp_${arch}),$(MAKE) -C $(lib_dsp_${arch}) dsp)
% endfor
% else:
	$(call build_if_exists,$(lib_dsp),$(MAKE) -C $(lib_dsp) dsp)
% endif

cpu_android: reg_android check_ndk
	$(call build_if_exists,$(lib_cpu),-$(MAKE) cpu_android -C $(lib_cpu))
	$(call build_if_exists,$(lib_cpu),@cp $(lib_cpu)/libs/aarch64-android/lib$(PACKAGE_NAME).so libs/arm64-v8a/libUdo$(PACKAGE_NAME)ImplCpu.so)

gpu_android: warn_gpu check_ndk reg_android
	$(call build_if_exists,$(lib_gpu),-$(MAKE) gpu_android -C $(lib_gpu))
	$(call build_if_exists,$(lib_gpu),@cp $(lib_gpu)/libs/aarch64-android/lib$(PACKAGE_NAME).so libs/arm64-v8a/libUdo$(PACKAGE_NAME)ImplGpu.so)

reg_android: check_ndk
	-$(ANDROID_NDK_ROOT)/ndk-build APP_MODULES="$(NDK_REG_LIB)" APP_ALLOW_MISSING_DEPS=true APP_ABI="$(PLATFORM)"

clean_android: check_ndk
	$(call build_if_exists,$(lib_cpu),-$(MAKE) clean_android -C $(lib_cpu))
	$(call build_if_exists,$(lib_gpu),-$(MAKE) clean -C $(lib_gpu))
	-$(ANDROID_NDK_ROOT)/ndk-build clean

# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))

warn_gpu:
ifneq (1,$(words [$(PLATFORM)]))
	$(warning WARNING: More than one platform selected for GPU compilation. libOpenCl is platform dependent and may not be compatible on all selected platforms.)
endif

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif

