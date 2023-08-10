# =============================================================================
#
#  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum

class Engine(Enum):
    SNPE = 'SNPE'
    ANN = 'ANN'
    QNN = 'QNN'


class Framework(Enum):
    tensorflow = 'tensorflow'
    tflite = 'tflite'
    onnx = 'onnx'


class Runtime(Enum):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    aip = 'aip'
    dspv65 = 'dspv65'
    dspv66 = 'dspv66'
    dspv68 = 'dspv68'
    dspv69 = 'dspv69'
    dspv73 = 'dspv73'
    aic = 'aic'

class Android_Architectures(Enum):
    aarch64_android = 'aarch64-android'
    aarch64_android_clang6_0 = 'aarch64-android-clang6.0'
    aarch64_android_clang8_0 = 'aarch64-android-clang8.0'

class X86_Architectures(Enum):
    x86_64_linux_clang = 'x86_64-linux-clang'

class Architecture_Target_Types(Enum):
    target_types = ['x86_64-linux-clang', 'aarch64-android']

class ComponentType(Enum):
    converter = "converter"
    context_binary_generator = "context_binary_generator"
    executor = "executor"
    inference_engine = "inference_engine"
    devices = "devices"
    snpe_quantizer = "snpe_quantizer"


class Status(Enum):
    off = "off"
    on = "on"
    always = "always"

class Devices_list(Enum):
    devices = ["linux-embedded", "android", "x86"]

class Device_type(Enum):
    linux_embedded = "linux-embedded"
    android = "android"
    x86 = "x86"
class AxisFormat(Enum):
    axis_format_mappings = {
        # (spatial-last format, spatial-first format): ((reshape order), (transpose order))
        ('NCDHW', 'NDHWC'): ((0, 4, 1, 2, 3), (0, 2, 3, 4, 1)),
        ('NCS', 'NSC'): ((0, 3, 1, 2), (0, 2, 3, 1)),
        ('NCF', 'NFC'): ((0, 2, 1), (0, 2, 1)),
        ('TNF', 'NTF'): ((1, 0, 2), (1, 0, 2)),
        ('IODHW', 'DHWIO'): ((2, 3, 4, 0, 1), (2, 3, 4, 0 ,1)),
        ('OIDHW', 'DHWOI'): ((3, 4, 0, 1, 2), (2, 3, 4, 1, 0)),
        ('OIDHW', 'DHWIO'): ((4, 3, 0, 1, 2), (0, 1, 2, 4, 3)),
        ('IOHW', 'HWIO'): ((2, 3, 0, 1), (2, 3, 0, 1)),
        ('OIHW', 'HWOI'): ((2, 3, 0, 1), (2, 3, 0, 1)),
        ('OIHW', 'HWIO'): ((3, 2, 0, 1), (2, 3, 1, 0))
    }
