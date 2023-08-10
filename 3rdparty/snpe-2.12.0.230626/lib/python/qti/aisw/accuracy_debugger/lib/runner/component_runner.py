# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import traceback
import pandas as pd

from qti.aisw.accuracy_debugger.lib.options.framework_diagnosis_cmd_options import FrameworkDiagnosisCmdOptions
from qti.aisw.accuracy_debugger.lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions
from qti.aisw.accuracy_debugger.lib.options.verification_cmd_options import VerificationCmdOptions
from qti.aisw.accuracy_debugger.lib.options.acc_debugger_cmd_options import AccDebuggerCmdOptions

from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_inference_engine_manager import InferenceEngineManager
from qti.aisw.accuracy_debugger.lib.verifier.nd_verification import Verification
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapping

from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError, InferenceEngineError, VerifierError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.utils.nd_symlink import symlink
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import retrieveQnnSdkDir, retrieveSnpeSdkDir


def exec_framework_diagnosis(args, logger=None):
    framework_args = FrameworkDiagnosisCmdOptions(args).parse()
    if (logger is None):
        logger = setup_logger(framework_args.verbose, framework_args.output_dir)

    logger.info(get_progress_message('PROGRESS_FRAMEWORK_STARTING'))

    symlink('latest', framework_args.output_dir, logger)

    try:
        framework_runner = FrameworkRunner(logger, framework_args)
        framework_runner.run()
        logger.info(get_progress_message('PROGRESS_FRAMEWORK_FINISHED'))
    except FrameworkError as e:
        raise FrameworkError("Conversion failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))

def exec_inference_engine(args, engine_type, logger=None):
    inference_engine_args = InferenceEngineCmdOptions(engine_type, args).parse()
    if (logger is None):
        logger = setup_logger(inference_engine_args.verbose, inference_engine_args.output_dir)

    # set engine path if none specified
    if engine_type == Engine.QNN.value and inference_engine_args.engine_path is None:
        inference_engine_args.engine_path = retrieveQnnSdkDir()
    elif engine_type == Engine.SNPE.value and inference_engine_args.engine_path is None:
        inference_engine_args.engine_path = retrieveSnpeSdkDir()

    logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_STARTING'))

    symlink('latest', inference_engine_args.output_dir, logger)

    try:
        inference_engine_manager = InferenceEngineManager(inference_engine_args, logger=logger)
        inference_engine_manager.run_inference_engine()
        get_mapping_arg = Namespace(None, framework=inference_engine_args.framework,
                                    version=inference_engine_args.framework_version, model_path=inference_engine_args.model_path,
                                    output_dir=inference_engine_args.output_dir, engine=inference_engine_args.engine,
                                    golden_dir_for_mapping=inference_engine_args.golden_dir_for_mapping)
        TensorMapping(get_mapping_arg, logger)
        logger.info(get_progress_message('PROGRESS_INFERENCE_ENGINE_FINISHED'))
    except InferenceEngineError as e:
        raise InferenceEngineError("Inference failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))

def exec_verification(args, logger=None):
    verification_args = VerificationCmdOptions(args).parse()
    if (logger is None):
        logger = setup_logger(verification_args.verbose, verification_args.output_dir)

    try:
        logger.info(get_progress_message("PROGRESS_VERIFICATION_STARTING"))
        if not verification_args.tensor_mapping:
            logger.warn("--tensor_mapping is not set, a tensor_mapping will be generated based on user input.")
            get_mapping_arg = Namespace(None, framework=verification_args.framework,
                                        version=verification_args.framework_version, model_path=verification_args.model_path,
                                        output_dir=verification_args.inference_results, engine=verification_args.engine,
                                        golden_dir_for_mapping=verification_args.framework_results)
            verification_args.tensor_mapping = TensorMapping(get_mapping_arg, logger)

        verify_results = []
        for verifier in verification_args.verify_types:
            verify_type = verifier[0]
            verifier_configs = verifier[1:]
            verification = Verification(verify_type, logger, verification_args, verifier_configs)
            if verification.has_specific_verifier() and len(verification_args.verify_types) > 1:
                raise VerifierError(get_message('ERROR_VERIFIER_USE_MULTI_VERIFY_AND_CONFIG'))
            verify_result = verification.verify_tensors()
            verify_result = verify_result.drop(columns=['Units', 'Verifier'])
            verify_result = verify_result.rename(columns={'Metric':verify_type})
            verify_results.append(verify_result)

        # if verification_args.verifier_config is None, all tensors use the same verifer. So we can export Summary
        if verification_args.verifier_config == None:
            summary_df = verify_results[0]
            for verify_result in verify_results[1:]:
                summary_df = pd.merge(summary_df, verify_result, on=['Name', 'LayerType', 'Size', 'Tensor_dims'])
            summary_df.to_csv(os.path.join(verification_args.output_dir, Verification.SUMMARY_NAME + ".csv"), index=False, encoding="utf-8")
            summary_df.to_html(os.path.join(verification_args.output_dir, Verification.SUMMARY_NAME + ".html"), index=False, classes='table')
        symlink('latest', verification_args.output_dir, logger)
        logger.info(get_progress_message("PROGRESS_VERIFICATION_FINISHED"))
    except VerifierError as excinfo:
        raise Exception("Verification failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))

def exec_wrapper(args, engine_type, logger=None):
    wrapper_args = AccDebuggerCmdOptions(args).parse()

    if (logger is None):
        logger = setup_logger(wrapper_args.verbose, wrapper_args.output_dir)

    # runs framework diagnosis
    exec_framework_diagnosis(args, logger=logger)

    # inference engine args pre-processing
    inference_args = list(args)
    model_name = os.path.basename(os.path.splitext(wrapper_args.model_path)[0])
    inference_args.extend(['--model_name', model_name])
    #replace --engine args to avoid ambiguity error
    if '--engine' in inference_args: inference_args[inference_args.index('--engine')] = '-e'

    # runs inference engine
    exec_inference_engine(inference_args, engine_type, logger=logger)

    # verification args pre-processing
    verification_args = list(args)
    graph_structure = model_name + '_graph_struct.json'
    graph_structure_path = os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest',
                                        graph_structure)
    verification_args.extend(['--graph_struct', graph_structure_path])

    verification_args.extend(['--inference_results', os.path.join(wrapper_args.working_dir,
                                                'inference_engine', 'latest','output/Result_0')])

    verification_args.extend(['--framework_results', os.path.join(wrapper_args.working_dir,
                                                    'framework_diagnosis', 'latest'),
                              '--tensor_mapping', os.path.join(wrapper_args.working_dir,
                                                    'inference_engine', 'latest', 'tensor_mapping.json')])

    if engine_type == Engine.QNN.value:
        qnn_model_net_json = model_name + '_net.json'
        qnn_model_net_json_path = os.path.join(wrapper_args.working_dir, 'inference_engine', 'latest', qnn_model_net_json)
        verification_args.extend(['--qnn_model_json_path', qnn_model_net_json_path])

    # runs verification
    exec_verification(verification_args, logger=logger)

    if engine_type == Engine.QNN.value and wrapper_args.deep_analyzer:
        # deep analyzer args pre-processing
        da_param_index = args.index('--deep_analyzer')
        deep_analyzers = args[da_param_index+1].split(',')
        del args[da_param_index:da_param_index+2]
        deep_analyzer_args = list(args)
        deep_analyzer_args.extend(['--tensor_mapping', os.path.join(wrapper_args.working_dir,
                                                        'inference_engine', 'latest', 'tensor_mapping.json'),
                                    '--inference_results', os.path.join(wrapper_args.working_dir,
                                                        'inference_engine', 'latest','output/Result_0'),
                                    '--graph_struct', graph_structure_path,
                                    '--framework_results', os.path.join(wrapper_args.working_dir,
                                                                'framework_diagnosis', 'latest'),
                                    '--result_csv', os.path.join(wrapper_args.working_dir,
                                                        'verification', 'latest','summary.csv')
                                                        ])
        # runs deep analyzers
        for d_analyzer in deep_analyzers:
            exec_deep_analyzer(deep_analyzer_args + ['--deep_analyzer',d_analyzer])


def exec_deep_analyzer(args, logger=None):
    da_args = AccuracyDeepAnalyzerCmdOptions(args).parse()
    if not os.path.isdir(da_args.output_dir):
        os.makedirs(da_args.output_dir)
    logger = setup_logger(da_args.verbose, da_args.output_dir)

    symlink('latest', da_args.output_dir, logger)

    try:
        from qti.aisw.accuracy_debugger.lib.deep_analyzer.nd_deep_analyzer import DeepAnalyzer
        from qti.aisw.accuracy_debugger.lib.options.accuracy_deep_analyzer_cmd_options import AccuracyDeepAnalyzerCmdOptions
        from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import DeepAnalyzerError

        if not da_args.tensor_mapping:
            logger.warn("--tensor_mapping is not set, a tensor_mapping will be generated based on user input.")
            get_mapping_arg = Namespace(None, framework=da_args.framework,
                                        version=da_args.framework_version, model_path=da_args.model_path,
                                        output_dir=da_args.inference_results, engine=da_args.engine,
                                        golden_dir_for_mapping=da_args.framework_results)
            da_args.tensor_mapping = TensorMapping(get_mapping_arg, logger)
        deep_analyzer = DeepAnalyzer(da_args, logger)
        deep_analyzer.analyze()
        logger.info("Successfully ran deep_analyzer!")
    except DeepAnalyzerError as excinfo:
        raise DeepAnalyzerError("deep analyzer failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))