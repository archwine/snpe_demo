# ==============================================================================
#
#  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import sys
import json
import glob
import csv
import copy
import shutil
import argparse
import subprocess
import numpy as np
import multiprocessing
import subprocess
import shlex
import time

import logging

from pathlib import Path

# setting logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT = '/snpe/'
# setting target compiler (eg ANDROID_NDK_ROOT)
MY_CWD = os.getcwd()
COMPILER = os.path.join(ROOT,'compiler')
os.environ["ANDROID_NDK_ROOT"] = COMPILER
sys.path.append(COMPILER)

SDK_PATH = os.environ.get("SNPE_ROOT")
if not SDK_PATH:
    SDK_PATH=os.path.join('/snpe','sdk')
    logger.error(
        "ERROR: Please set environment before running this script. Follow steps mentioned in README")
    exit(1)

# including files present in benchmark folder sdk
sys.path.append(os.path.join(SDK_PATH, "benchmarks/"))


from common_utils.constants import LOG_FORMAT
from common_utils import exceptions
from snpebm import snpebm_constants

from common_utils.protocol.adb import Adb
from common_utils.device import Device
from common_utils.common import execute

TNR_TEST_TIMEOUT = 300

class Converter(object):

    def __init__(self, config):
        self.cfg = config
        self.user_data = '/snpe/user_data' if os.path.isdir('/snpe/user_data') else '/snpe/output'

    def convert(self, model, output_path):
        """Convert the model to dlc"""

        if not model:
            logging.error("Invalid model %s received!", model)
            exit(1)

        cfg = self.cfg
        model_cfg =  cfg['Model']
        converter_cfg = model_cfg['Conversion']
        cmd = converter_cfg['Command']

        cmd_args = shlex.split(cmd)
        logging.debug("Original cmd: %s", cmd)
        converted_model_path = os.path.join(output_path,"model.dlc")
        found_graph = False
        found_output = False
        for i,k in enumerate(cmd_args):
            # Check for the graph definition
            if '--input_network' == k or '-i' == k:
                cmd_args[i+1] = model
                found_graph = True

            if '--output' == k:
                cmd_args[i+1] = converted_model_path


        if found_graph == False:
            cmd_args.append('-i')
            cmd_args.append(model)

        if found_output == False:
            cmd_args.append('--output')
            cmd_args.append(converted_model_path)


        logging.info("Modified converter cmd: %s", cmd_args)
        logging.info("*******************************************************************")
        logging.info("Converting received model:\n%s\n", cmd)
        result = subprocess.run(cmd_args, cwd=self.user_data, check=True) #, capture_output=True, text=True, check=True)
        logging.debug("Model conversion output %s", result.stdout)
        if result.returncode != 0:
            logging.error("Model conversion failure w/error %s", result.stderr)
            return ""

        logging.info("DLC generated: ")
        logging.info("%s",converted_model_path)
        logging.info("*******************************************************************\n")

        logging.info("stdout: %s", result.stdout)
        logging.info("stderr: %s", result.stderr)
        return converted_model_path


    def quantize(self, model, output_path, backend='DSP'):

        # If a previous failure occurred there won't be a valid model
        if not model:
            return ""

        p_model = Path(model)
        quantized_model = str(p_model.with_suffix(''))+'_quantized.dlc'

        model_cfg =  self.cfg['Model']
        cmd = model_cfg['Quantization']['Command']
        cmd_args = shlex.split(cmd)

        if 'InputList'not in model_cfg:
            logging.error("Must pass InputList arg for DSP backend quantization")
            exit(1)
        else:
            input_list = os.path.join(self.user_data, model_cfg['InputList'])

        input_list_found = False
        for i,k in enumerate(cmd_args):
            # Check for the graph definition
            if '--input_list' == k:
                cmd_args[i+1] = input_list
                input_list_found = True

        logging.debug("Original cmd: %s", cmd)
        found_graph = False
        found_output = False
        for i,k in enumerate(cmd_args):
            # Check for the graph definition
            if '--input_dlc' == k or '-i' == k:
                cmd_args[i+1] = model
                found_graph = True

            if '--output_dlc' == k:
                cmd_args[i+1] = quantized_model


        if not found_graph:
            cmd_args.append('--input_dlc')
            cmd_args.append(model)

        if not found_output:
            cmd_args.append('--output_dlc')
            cmd_args.append(quantized_model)

        if not input_list_found:
             cmd_args.append('--input_list')
             cmd_args.append(input_list)


        logging.info("Modified quantizer cmd: %s", cmd_args)
        logging.info("*******************************************************************")

        result = subprocess.run(cmd_args, cwd=self.user_data, check=True)
        if result.returncode != 0:
            logging.error("Model quantization failure w/error %s", result.stderr)
            return ""

        logging.info("Quantized model available here %s\n", quantized_model);
        logging.info("*******************************************************************\n");

        return quantized_model


class OutputVerifier(object):
    def __init__(self):
        self.checker_identifier = 'basic_verifier'
        pass

    def verify_output(self, inputs_dir, outputs_dir, expected_outputs_dir,
                      sanity=False, num_of_batches=1, output_data_type=None):
        if sanity:
            input_list_path = 'input_list_sanity.txt'
        else:
            input_list_path = 'input_list.txt'
        with open(os.path.join(inputs_dir, input_list_path)) as inputs_list_file:
            inputs_list = inputs_list_file.readlines()
            for inputs in inputs_list:
                if inputs.startswith('#'):
                    inputs_list.remove(inputs)

            # verify if the number of results is the same missing
            number_of_checks_passing = 0
            iterations_verification_info_list = []
            failure_list = []
            for iteration in range(0, len(inputs_list)):
                iteration_input_files = (
                    inputs_list[iteration].strip()).split(' ')
                iteration_input_files = [input_file.split(
                    ':=')[-1] for input_file in iteration_input_files]
                iteration_input_files = [
                    os.path.join(
                        inputs_dir,
                        f) for f in iteration_input_files]
                iteration_result_str = 'Result_' + str(iteration)
                is_passing, iterations_verification_info, failures = self._verify_iteration_output(
                    iteration_input_files,
                    outputs_dir,
                    expected_outputs_dir,
                    iteration_result_str,
                    num_of_batches,
                    output_data_type
                )

                if is_passing:
                    number_of_checks_passing += 1
                else:
                    failure_list.extend(failures)
                iterations_verification_info_list += iterations_verification_info

            return number_of_checks_passing, \
                   len(inputs_list), \
                   iterations_verification_info_list, \
                   failure_list

    def _verify_iteration_output(self, input_files, outputs_dir, expected_dir, iteration_result_str,
                                 num_of_batches=1, output_data_type=None):
        is_passing = True
        iterations_verification_info_list = []
        failure_list = []
        for root, dirnames, filenames in os.walk(os.path.join(expected_dir, iteration_result_str)):
            for output_layer in filenames:
                if "linux" in sys.platform:
                    output_result = root.split(iteration_result_str)[1].replace("/", "_")[1:]
                else:
                    output_result = root.split(iteration_result_str)[1].replace("\\", "_")[1:]
                if output_result:
                    output_name = output_result.replace(":", "_") + "_" + \
                                  output_layer.replace("-", "_").replace(":", "_")
                else:
                    output_name = output_layer.replace("-", "_").replace(":", "_")

                if output_data_type is not None:
                    output = output_name.split('.')
                    output_name = output[0] + "_" + \
                                  output_data_type.split('_')[0] + "." + output[1]
                expected_output_file = os.path.join(root, output_layer)

                output_file = os.path.join(outputs_dir, iteration_result_str, output_name)

                # hack around for splitter - needs to be removed
                # in the future
                if not os.path.exists(expected_output_file):
                    expected_output_file_intermediate = os.path.join(
                        expected_dir,
                        output_file[output_file.find('Result_'):]).split(iteration_result_str)

                    expected_output_file = expected_output_file_intermediate[0] \
                                           + iteration_result_str + '/' + \
                                           expected_output_file_intermediate[1]
                # end of hack
                # changing output name if raw file name starts with integer
                if not (os.path.exists(output_file)) and output_name[0].isdigit():
                    output_name = '_' + output_name
                    output_file = os.path.join(
                        outputs_dir,
                        iteration_result_str,
                        output_name
                    )
                iteration_verification_info = self._verify_iteration_output_file(
                    input_files,
                    output_file,
                    expected_output_file,
                    num_of_batches
                )
                iterations_verification_info_list.append(
                    iteration_verification_info[1])
                if not iteration_verification_info[0]:
                    is_passing = False
                    failure_list.append(expected_output_file)
        return is_passing, iterations_verification_info_list, failure_list

    def _verify_iteration_output_file(self, input_files, output_file, expected_output_file,
                                      num_of_batches=1):
        try:
            output = np.fromfile(output_file, dtype=np.float32)
            expected_output = np.fromfile(
                expected_output_file, dtype=np.float32)
        except IOError:
            raise Exception('Can not open the golden or predicted files (names does not match). '
                            'Please check both the output and golden directories contain: %s'
                            % os.path.basename(output_file))

        result = self._verify_iteration_output_helper(input_files, output, expected_output,
                                                      num_of_batches)
        if not result[0]:
            logger.error(
                'Failed to verify %s on %s' % (expected_output_file, self.checker_identifier))
        return result

    def _verify_iteration_output_helper(self, input_files, output, expected_output, num_of_batches):
        pass


class CosineSimilarity(OutputVerifier):
    def __init__(self, threshold = 0.9):
        super(CosineSimilarity, self).__init__()
        self.checker_identifier = 'CosineSimilarity'
        self.threshold = threshold

    def _verify_iteration_output_helper(
            self, input_files, output, expected_output, num_of_batches=1):
        output, expected_output = output.flatten(), expected_output.flatten()
        num = output.dot(expected_output.T)
        denom = np.linalg.norm(output) * np.linalg.norm(expected_output)
        if denom == 0:
            return [False, False]
        else:
            similarity_score = num / denom
            if similarity_score >= self.threshold:
                return [True, True]
            else:
                return [False, False]


class RtolAtolOutputVerifier(OutputVerifier):
    def __init__(self, rtolmargin=1e-2, atolmargin=1e-2):
        super(RtolAtolOutputVerifier, self).__init__()
        self.checker_identifier = 'RtolAtolVerifier_w_r_' + \
                                  str(rtolmargin) + '_a_' + str(atolmargin)
        self.rtolmargin = rtolmargin
        self.atolmargin = atolmargin

    def _calculate_margins(self, expected_output):
        return self.atolmargin, self.rtolmargin

    def _verify_iteration_output_helper(
            self, input_files, output, expected_output, num_of_batches=1):
        adjustedatolmargin, adjustedrtolmargin = self._calculate_margins(
            expected_output)
        match_array = np.isclose(
            output,
            expected_output,
            atol=adjustedatolmargin,
            rtol=adjustedrtolmargin)

        notclose = (len(match_array) - np.sum(match_array))
        if notclose == 0:
            return [True, False]
        else:
            return [False, False]


class RunTestPackage(object):

    def runner(self):
        pass

    @staticmethod
    def _run_command(cmd):
        try:
            logger.debug("Running - {}".format(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            out, err = process.communicate()
            return_code = process.returncode
            return out, err, return_code
        except Exception as err:
            logger.error("Error Observed: {}".format(err))
            return "", "Error", "1"

    @staticmethod
    def check_models(user_model):
        """
            This function will check model name.
            By default model_run will have all models.
        """

        if not user_model:
            model_run = MODELS
        elif isinstance(user_model, str):
            if user_model.lower() == "all":
                model_run = MODELS
            else:
                model_run = user_model.split(" ")
        else:
            model_run = MODELS
        logger.debug("Running test for {}".format(model_run))
        return model_run

    @staticmethod
    def check_dsp_type(chipset):
        """
        :description: This fucntion will return dsp type based on soc.
        :return: dsp_type
        """
        # add more mapping if required
        dsp_type = None
        if chipset in ["8350", "7350", "7325", "lahaina", "cedros", "kodiak"]:
            dsp_type = 'v68'
        elif chipset in ["8450", "waipio", "7450", '8475', 'fillmore', 'palima', '8415', 'alakai']:
            dsp_type = 'v69'
        elif chipset in ["6450", "netrani"]:
            dsp_type = 'v69-plus'
        elif chipset in ["kailua", '8550', '1230', 'sxr1230', '2115', 'ssg2115']:
            dsp_type = 'v73'
        elif chipset in ["6375", "4350", "strait", "mannar", "5165", 'qrb5165', '610', 'qcs610']:
            dsp_type = 'v66'
        else:
            logger.error("Please provide --dsp_type argument value")
            exit(1)
        return dsp_type

    @staticmethod
    def check_mem_type(chipset):
        """
        :description: This function will return mem type based on soc.
        :return: mem_type
        """
        # add more mapping if required
        mem_type = None
        if chipset in ['8450', 'waipio', 'kailua', '8550', '8475', 'palima', '8415', 'alakai']:
            mem_type = '8m'
        elif chipset in ["8350", "lahaina"]:
            mem_type = '4m'
        elif chipset in ['7350', 'cedros', '7325', 'kodiak', '7450', 'fillmore', '1230', \
                         'sxr1230', '2115', 'ssg2115', "6450", "netrani"]:
            mem_type = '2m'

        return mem_type

    @staticmethod
    def check_toolchain(chipset):
        """
        :description: This function will return toolchain based on soc.
        :return: toolchain
        """
        # add more mapping if required
        mem_type = None
        if chipset in ['5165', 'qrb5165', '1230', 'sxr1230']:
            toolchain = 'aarch64-oe-linux-gcc9.3'
        elif chipset in ['610', 'qcs610']:
            toolchain = 'aarch64-oe-linux-gcc8.2'
        else:
            toolchain = 'aarch64-android-clang8.0'
        return toolchain

    @staticmethod
    def copy_sdk_artifacts(tmp_work_dir, dsp_type, toolchain):
        artifacts = {}
        # first check for ZDL_ROOT, because if you have both, contributor role takes precedence
        _sdk_root = SDK_PATH
        lib_dir = os.path.join(tmp_work_dir, toolchain, 'lib')
        bin_dir = os.path.join(tmp_work_dir, toolchain, 'bin')
        os.makedirs(lib_dir, exist_ok = True)
        os.makedirs(bin_dir, exist_ok = True)
        if snpebm_constants.ZDL_ROOT not in os.environ:
            if os.path.exists(os.path.join(_sdk_root, "lib", toolchain)):
                artifacts['lib'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_sdk_root, "lib", toolchain)), x),os.listdir(os.path.join(_sdk_root, "lib", toolchain))))
            if os.path.exists(os.path.join(_sdk_root, "lib", "dsp")):
                artifacts['dsp'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_sdk_root, "lib", "dsp")), x),os.listdir(os.path.join(_sdk_root, "lib", "dsp"))))
            if os.path.exists(os.path.join(_sdk_root, "bin", toolchain)):
                artifacts['bin'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_sdk_root, "bin", toolchain)), x),os.listdir(os.path.join(_sdk_root, "bin", toolchain))))
        else:
            _zdl_root = SDK_PATH
            if os.path.exists(os.path.join(_zdl_root, toolchain, "lib")):
                artifacts['lib'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_zdl_root, toolchain, "lib")), x),os.listdir(os.path.join(_zdl_root, toolchain, "lib"))))
            if os.path.exists(os.path.join(_zdl_root, toolchain, "lib", "dsp")):
                artifacts['dsp'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_zdl_root, toolchain, "lib", "dsp")), x),os.listdir(os.path.join(_zdl_root, toolchain, "lib", "dsp"))))
            if os.path.exists(os.path.join(_zdl_root, toolchain, "bin")):
                artifacts['bin'] = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(_zdl_root, toolchain, "bin")), x),os.listdir(os.path.join(_zdl_root, toolchain, "bin"))))

        for lib in artifacts['lib']:
            shutil.copy(lib, lib_dir)
        for lib in artifacts['dsp']:
            shutil.copy(lib, lib_dir)
        for bin in artifacts['bin']:
            shutil.copy(bin, bin_dir)

        return

    @staticmethod
    def push_to_device(adb, inputs, on_device_path):
        ret = True
#        logger.info('Pushing artifacts {} to device: {}'.format(inputs, adb.device))
        for inp in inputs:
            if os.path.isdir(inp):
                logger.info("Pushing {} to {} on device {}".format(inp, on_device_path, adb._adb_device))
            else:
                logger.debug(os.path.join(on_device_path, os.path.basename(inp)))
                adb.shell('rm -rf {0}'.format(os.path.join(on_device_path, os.path.basename(inp))))
                logger.debug('Existing test directory removed from device.')
            return_code, out, err = adb.push(inp, on_device_path)
            if return_code:
                ret = False
                logger.error("ADB Push Failed!!")
                break
        return ret

    @staticmethod
    def pull_from_device(adb, on_device_inputs, host_dest_path):
        ret = True
        for inp in on_device_inputs:
            return_code, out, err = adb.pull(inp, host_dest_path)
            if return_code:
                ret = False
                logger.error("ADB Pull Failed!!")
                break
        return ret

    def check_deviceid(self, device, hostname):
        """
        :description:
            This function will fetch the devices connected to local host.
        :return:
            device id
        """
        if not device and hostname == "localhost":
            cmd = "adb devices | awk '{print $1;}'"
            out, err, code = self._run_command(cmd)
            if code or err:
                logger.error("Error Observed: {}".format(err))
                logger.error("Check if device is detectable and provide device id as argument")
                return

            if not isinstance(out, str):
                out = out.decode('utf-8')
            devices = out.split("\n")
            device_id = ""
            if devices[1]:
                device_id = devices[1]
            return device_id

        elif not device and not hostname == "localhost":
            logger.error("Please provide device id")
            exit(1)
        else:
            return device


class RunBenchmark(RunTestPackage):

    def __init__(self, args, latency_worker_id):

        self.converter = Converter(args)
        self.local_path=args["HostRootPath"]
        self.device_path=args['DevicePath']
        self.deviceId = self.check_deviceid(args['Devices'][latency_worker_id], 'localhost')
        self.chipset =  args['Chipset'] if 'Chipset' in args else None
        self.dsp_type = self.check_dsp_type(self.chipset)
        self.mem_type = self.check_mem_type(self.chipset)
        self.toolchain = self.check_toolchain(self.chipset)
        adb = Adb('adb', self.deviceId)
        adb._execute('shell', ['mkdir', '-p', self.device_path])
        if self.chipset in ['8550', '5165', '610', 'qrb5165', 'qcs610', '1230', 'sxr1230', '2115', 'ssg2115']:
            adb._execute('root', [])

        self.device = Device(self.deviceId, [self.toolchain], self.device_path)
        self.device.init_env(self.device_path, False)
        self.artifacts=os.path.join('/tmp', self.deviceId)
        if os.path.exists(self.artifacts):
            shutil.rmtree(self.artifacts)
        os.makedirs(self.artifacts)
        self._setup_env()
        self.copy_sdk_artifacts(self.artifacts, self.dsp_type, self.toolchain)

        self.hostname = 'localhost'
        logger.debug("Device id on which test will run: {}".format(self.deviceId))

        self.backend = args['Backends']

        self.itr = 1 #int(args.iterations)
        self.verify = False
        self.output_file_name = "benchmark_"
        self.perf = 'perf'
        self.shared_buffer = False #args.shared_buffer

        self.input_list = args["Model"]["InputList"] if os.path.isabs(args["Model"]["InputList"]) else os.path.join(self.local_path, args["Model"]["InputList"])
        self.inputs = args["Model"]["Data"] if os.path.isabs(args["Model"]["Data"]) else os.path.join(self.local_path, args["Model"]["Data"])
        self.device.push_data(os.path.join(self.artifacts, self.toolchain), self.device_path)
        self.device.push_data(self.input_list, self.device_path)
        self.device.push_data(self.inputs, self.device_path)

        return

    def _setup_env(self):
#        logger.debug(           '[{}] Pushing envsetup scripts'.format(                self._adb._adb_device))
        lib_dir = os.path.join(self.device_path, self.toolchain, 'lib')
        bin_dir = os.path.join(self.device_path, self.toolchain, 'bin')

        commands = [
            'export LD_LIBRARY_PATH={}:/vendor/lib64/:$LD_LIBRARY_PATH'.format(lib_dir),
            'export ADSP_LIBRARY_PATH="{};/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/dsp;/usr/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/etc/images/dsp;"'.format(
                lib_dir),
            'export PATH={}:$PATH'.format(bin_dir)
        ]
        script_name = os.path.join(self.artifacts, '{}_{}_{}'.format(self.deviceId, self.toolchain, 'env.sh'))
        with open(script_name, 'w') as f:
            f.write('\n'.join(commands) + '\n')
        self.device.push_data(script_name, self.device_path)
           # self._adb.push(script_name, self._device_root)
        self.env_script = os.path.join(self.device_path, os.path.basename(script_name))

    def execute_bm(self, model):

        output_dir = os.path.join(self.device_path, "output")
        cmd  = "source {} && ".format(self.env_script)
        cmd += "cd {} && ".format(self.device_path)
        cmd += "{} ".format("snpe-net-run")
        cmd += "{} ".format(snpebm_constants.RUNTIMES[self.backend])
        cmd += "--input_list {} ".format(os.path.join(self.device_path, os.path.basename(self.input_list)))
        cmd += "--profiling_level {} ".format("basic")
        cmd += "--perf_profile {} ".format("burst")
        cmd += "--output_dir {} ".format(output_dir)
        cmd += "--container {} ".format(os.path.join(self.device_path, os.path.basename(model)))
        logging.info("\nRunning benchmark:\n{}\n".format(cmd))
        return_code, output, err_msg = self.device.device_helper._execute('shell', shlex.split(cmd))
        if return_code != 0:
            logging.error("Benchmark FAILED:\n{}".format(err_msg))
            exit(1)

        logging.info("Benchmark SUCCESS:\n{}".format(output))
        logging.info("Copying results from {} to {}".format(output_dir, os.path.dirname(model)))
        self.device.device_helper.pull(output_dir, self.current_hil_path)
        return True

    def convert(self, model, output_path):
        self.current_hil_path = output_path
        model = self.converter.convert(model, output_path)
        if self.backend == 'DSP':
            model = self.converter.quantize(model, output_path)
        return model

    def runner(self, model):
        # running benchmark
        logging.info("*******************************************************************\n");
        logging.info("Preparing to run inference, pushing model")
        self.device.push_data(model, self.device_path)

        self.execute_bm(model)
        stats = self.process_results()
        logging.info("*******************************************************************\n");
        return stats


    def process_results(self):
        stats = {}
        profiling_file = os.path.join(self.current_hil_path, 'output/SNPEDiag.log')
        if not os.path.exists(profiling_file):
            logging.error("Invalid profiling file {}".format(profiling_file))
            exit(1)

        parsed_file = str(Path(profiling_file).with_suffix('.txt'))
        cmd = SDK_PATH+'/bin/x86_64-linux-clang/snpe-diagview'
        cmd += ' --input_log ' + profiling_file # + '  > ' + parsed_file

        logging.info("*******************************************************************");
        logging.info("Parsing stats %s:\n", profiling_file);

        cmd_args = shlex.split(cmd)
        result = subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            logging.error("Couldn't process statistics result file {}".format(profiling_file));
            exit(1)

        print(str(result.stdout))
        text_file = open(parsed_file, "wt")
        text_file.write(result.stdout)
        text_file.close()

        with open(parsed_file, 'r') as hdl:
            log_file = hdl.readlines()

        execute_time = 0
        for data in log_file:
            statistic_key = data.split(": ")[0]
            if statistic_key == "Total Inference Time":
                execute_time = int(data.split(": ")[1].partition(' us')[0])
                break

        #execute_time = exec_times['Backend (u accelerator (execute) time)']
        logger.info("-----------------------------------------------------------")
        logger.info("[SNPE Backend Execute Time (Avg)]: {} us".format(execute_time))
        logger.info("-----------------------------------------------------------")

        # Return SNPE backend latency (in microseconds)
        stats = { 'latency_in_us' : execute_time, 'model_memory' : 0.0 }
        return stats
