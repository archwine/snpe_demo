# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Runtime, Framework
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from packaging import version

import argparse

class LayerwiseSnoopingCmdOptions(CmdOptions):

    def __init__(self, args):
        super().__init__('layerwise_snooping', args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to run layerwise_snooping."
        )

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-m', '--model_path', type=str, required=True,
                            help='path to original model that needs to be dissected.')
        required.add_argument('--snooping', type=str, required=True,
                            choices=['layerwise','fp16sweep'], #'referencecodeAnalyzer' will be supported in future
                            help='Snooping performs debugging at layer level')
        required.add_argument('--verifier',type=str, default='mse',
                                 help='comma delimited verifiers to be used. e.g '
                                      'avg,rme')
        required.add_argument('--framework_results', type=str, required=True,
                            help='Path to root directory generated from framework diagnosis. '
                                'Paths may be absolute, or relative to the working directory.')
        required.add_argument('-f', '--framework', nargs='+', type=str, required=True,
                            help='Framework type to be used, followed optionally by framework '
                                'version.')
        required.add_argument('-e', '--engine', nargs='+', type=str, required=True,
                            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
                            help='Name of engine that will be running inference, '
                                'optionally followed by the engine version.')
        required.add_argument('-p', '--engine_path', type=str, required=True,
                                help='Path to the inference engine.')
        required.add_argument('-l', '--input_list', type=str, required=True,
                                help="Path to the input list text.")

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument('-r', '--runtime', type=str.lower, default=Runtime.dspv68.value,
                                choices=[r.value for r in Runtime], help="Runtime to be used.")
        optional.add_argument('-t', '--target_device', type=str.lower, default='x86',
                                choices=['x86', 'android', 'linux-embedded'],
                                help='The device that will be running inference.')
        optional.add_argument('-a', '--architecture', type=str.lower, default='x86_64-linux-clang',
                                choices=['x86_64-linux-clang', 'aarch64-android'],
                                help='Name of the architecture to use for inference engine.')
        optional.add_argument('--deviceId', required=False, default=None,
                             help='The serial number of the device to use. If not available, '
                                 'the first in a list of queried devices will be used for validation.')
        optional.add_argument('--result_csv', type=str, required=False,
                                help='Path to the csv summary report comparing the inference vs framework'
                                'Paths may be absolute, or relative to the working directory.'
                                'if not specified, then a --problem_inference_tensor must be specified')
        optional.add_argument('--verifier_threshold', type=float, default=None,
                            help='Verifier threshold for problematic tensor to be chosen.')
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dir', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist.')
        optional.add_argument('--verifier_config', type=str, default=None, help='Path to the verifiers\' config file')
        optional.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='Verbose printing')
        optional.add_argument('--start_layer', type=str, default=None, required=False,
                                    help="Extracts the given model from mentioned start layer output name")
        optional.add_argument('--end_layer', type=str, default=None, required=False,
                                    help="Extracts the given model from mentioned end layer output name")
        optional.add_argument('--precision', choices=['int8', 'fp16'], default='int8',
                             help='select precision')
        optional.add_argument('--compiler_config', type=str, default=None, required=False,
                                    help="Path to the compiler config file.")
        optional.add_argument('-bbw', '--bias_bitwidth', type=int, required=False, default=8, choices=[8, 32],
                                help="option to select the bitwidth to use when quantizing the bias. default 8")
        optional.add_argument('-abw', '--act_bitwidth', type=int, required=False, default=8, choices=[8, 16],
                                help="option to select the bitwidth to use when quantizing the activations. default 8")
        optional.add_argument('-wbw', '--weights_bitwidth', type=int, required=False, default=8, choices=[8],
                                help="option to select the bitwidth to use when quantizing the weights. Only support 8 atm")
        optional.add_argument('-pq', '--param_quantizer', type=str.lower, required=False, default='tf',
                                        choices=['tf','enhanced','adjusted','symmetric'],
                                        help="Param quantizer algorithm used.")

        optional.add_argument('-qo', '--quantization_overrides', type=str, required=False, default=None,
                                    help="Path to quantization overrides json file.")

        optional.add_argument('--act_quantizer', type=str, required=False, default='tf',
                                    choices=['tf','enhanced','adjusted','symmetric'],
                                    help="Optional parameter to indicate the activation quantizer to use")

        optional.add_argument('--algorithms', type=str, required=False, default=None,
                                    help="Use this option to enable new optimization algorithms. Usage is: --algorithms <algo_name1> ... \
                                        The available optimization algorithms are: 'cle ' - Cross layer equalization includes a number of methods for \
                                        equalizing weights and biases across layers in order to rectify imbalances that cause quantization errors.\
                                        and bc - Bias correction adjusts biases to offse activation quantization errors. Typically used in \
                                        conjunction with cle to improve quantization accuracy.")

        optional.add_argument('--ignore_encodings', action="store_true", default=False,
                                    help="Use only quantizer generated encodings, ignoring any user or model provided encodings.")

        optional.add_argument('--per_channel_quantization', action="store_true", default=False,
                                    help="Use per-channel quantization for convolution-based op weights.")
        optional.add_argument('--extra_converter_args', type=str, required=False, default=None,
                                    help="additional convereter arguments in a string. \
                                          example: --extra_converter_args input_dtype=data float;input_layout=data1 NCHW")
        optional.add_argument('--extra_runtime_args', type=str, required=False, default=None,
                                    help="additional convereter arguments in a quoted string. \
                                        example: --extra_runtime_args profiling_level=basic;log_level=debug")

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.result_csv = get_absolute_path(parsed_args.result_csv)
        parsed_args.framework_results = get_absolute_path(parsed_args.framework_results)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)
        parsed_args.compiler_config = get_absolute_path(parsed_args.compiler_config)

        if parsed_args.framework[0] == 'onnx':
            onnx_env_available = False
            try:
                import onnx
                onnx_env_available = True
            except :
                pass
            if onnx_env_available and version.parse(onnx.__version__) <= version.parse('1.8.0'):
                raise UnsupportedError("Layerwise snooping requires onnx version >= 1.8.0")

        # get engine and engine version if possible
        parsed_args.engine_version = None
        if len(parsed_args.engine) > 2:
            raise ParameterError("Maximum two arguments required for inference engine.")
        elif len(parsed_args.engine) == 2:
            parsed_args.engine_version = parsed_args.engine[1]

        parsed_args.engine = parsed_args.engine[0]
        if parsed_args.engine != 'QNN':
            raise UnsupportedError("Layerwise snooping supports only QNN")

        # get framework and framework version if possible
        parsed_args.framework_version = None
        if len(parsed_args.framework) > 2:
            raise ParameterError("Maximum two arguments required for framework.")
        elif len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]

        parsed_args.framework = parsed_args.framework[0]
        if parsed_args.framework != 'onnx':
            raise UnsupportedError("Layerwise snooping supports only onnx framework")

        # verify that target_device and architecture align
        arch = parsed_args.architecture
        linux_target, android_target = (parsed_args.target_device == 'x86' or parsed_args.target_device == 'linux_embedded'), parsed_args.target_device == 'android'
        if linux_target and parsed_args.runtime==Runtime.dspv66.value: raise ParameterError("Engine and runtime mismatch.")
        linux_arch = android_arch = None
        if parsed_args.engine == Engine.SNPE.value:
            linux_arch, android_arch = arch == 'x86_64-linux-clang', arch.startswith('aarch64-android-clang')
            if parsed_args.runtime not in ["cpu","dsp","gpu","aic"]:
                raise ParameterError("Engine and runtime mismatch.")
        else:
            linux_arch, android_arch = arch == 'x86_64-linux-clang', arch == 'aarch64-android'
            if parsed_args.runtime not in ["cpu","dsp","dspv66","dspv68","dspv69","gpu","aic"]:
                raise ParameterError("Engine and runtime mismatch.")
            dspArchs=[r.value for r in Runtime if r.value.startswith("dsp") and r.value != "dsp"]
            if parsed_args.runtime == "dsp": parsed_args.runtime=max(dspArchs)
        if not ((linux_target and linux_arch) or (android_target and android_arch)):
            raise ParameterError("Target device and architecture mismatch.")

        return parsed_args
