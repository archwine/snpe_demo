# =============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import os
import sys

try:
    from qti.aisw.dlc_utils import modeltools
except ImportError as ie1:
    print("Failed to find necessary python package")
    print(str(ie1))
    print("Please ensure that libDlModelToolsPy3.so is discoverable your PYTHONPATH")
    sys.exit(1)

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.backend_base import BackendTranslationBase
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.translation_utils import get_si_notation
from qti.aisw.converters.qnn_backend.qnn_translations import QnnTranslations
from qti.aisw.converters.qnn_backend.qnn_backend_base import QnnConverterBackendBase
from qti.aisw.converters.qnn_backend.qnn_mappings import *
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
from qti.aisw.converters.backend.custom_ops.op_factory import UDOFactory


# TODO: updated inheritance to ConverterBackend once alignment of Ops are complete
class DLCBackend(QnnConverterBackendBase):
    class ArgParser(QnnConverterBackendBase.ArgParser):
        def __init__(self, **kwargs):
            super(DLCBackend.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument('--model_version', type=str, default=None,
                                       help='User-defined ASCII string to identify the model, only first '
                                            '64 bytes will be stored')
            self.add_optional_argument('--validation_target', nargs=2,
                                       action=validation_utils.ValidateTargetArgs,
                                       help="A combination of processor and runtime target against which model "
                                            "will be validated. \n"
                                            "Choices for RUNTIME_TARGET: \n   {cpu, gpu, dsp}. \n"
                                            "Choices for PROCESSOR_TARGET: \n"
                                            "   {snapdragon_801, snapdragon_820, snapdragon_835}.\n"
                                            "If not specified, will validate model against "
                                            "{snapdragon_820, snapdragon_835} across all runtime targets.",
                                       metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'),
                                       default=[], )
            self.add_optional_argument('--strict', dest="enable_strict_validation",
                                       action="store_true",
                                       default=False,
                                       help="If specified, will validate in strict mode whereby model will not "
                                            "be produced if it violates constraints of the specified validation "
                                            "target. If not specified, will validate model in permissive mode "
                                            "against the specified validation target.")
            self.add_optional_argument("--udo_config_paths", "-udo", nargs='+',
                                       dest="custom_op_config_paths",
                                       action=validation_utils.check_json(),
                                       help="Path to the UDO configs (space separated, if multiple)")

    def __init__(self, args):
        super(DLCBackend, self).__init__(args)
        # get converter args for saving dlc
        if self.output_model_path is None:
            filename, _ = os.path.splitext(os.path.realpath(self.input_model_path))
            self.output_path = filename + ".dlc"
        else:
            self.output_path = self.output_model_path
        self.model_version = args.model_version
        self.validation_target = args.validation_target
        self.enable_strict_validation = args.enable_strict_validation
        self.serialize_with_suppl_attr = True

        # Ensure model version fits in 64 bytes to match dlcv3
        model_version = self.model_version
        if model_version:
            model_version = model_version[:64]
        else:
            model_version = ''

        self.dlc_serializer = modeltools.IrDlcSerializer(self.output_path,
                                                         self.copyright_str,
                                                         model_version,
                                                         self.converter_command)

    # TODO: Cleanup when all ops are aligned to QNN
    """ Start of clean up """
    def add_tensor(self, node_name, tensor_name, tensor_type, tensor: np.ndarray,
                   check_encodings=True, tensor_data_type=ir_graph.QNN_DATATYPE_FLOAT_32,
                   src_axis_format=None, tensor_axis_format=None, orig_tensor_name=None):
        data = None
        if tensor_type == qnn_definitions.QNN_TENSOR_TYPE_STATIC:
            data = tensor
        tensor_info = self.create_tensor_info(tensor_name, tensor_type, tensor.shape,
                                              tensor_data_type, src_axis_format, tensor_axis_format, data=data,
                                              encoding=None)

        is_quantizable = True
        if tensor_data_type != ir_graph.QNN_DATATYPE_FLOAT_32 or not check_encodings:
            is_quantizable = False

        if not self.model.add_tensor(node_name, tensor_info, is_quantizable=is_quantizable):
            raise RuntimeError("Adding Tensor {} for Node {} failed.".format(node_name, tensor_name))

    def add_node(self, node_name, node_type, input_names, outputs_info, tensor_params={}, scalar_params={},
                 macs=0):
        # resolve package names for each node name
        node_package_name = self.resolve_package_names(node_type)

        if not self.model.add_node(node_name, node_type, node_package_name, tensor_params, scalar_params,
                                   input_names, outputs_info):
            raise RuntimeError("Adding Node {} failed.".format(node_name))

    @staticmethod
    def sanitize_name(name):
        return name

    @staticmethod
    def _sanitize_tensor_name(tensor_name):
        return tensor_name

    """ End of clean up """

    # overrides the set_package_dict method in qnn_backend_base
    # to correctly set the package dict info for snpe 2.0 udo
    def set_package_dict(self, graph):
        if self.package_name:
            package_name_dict = {self.package_name: [node.op.type for node in graph.list_nodes()[1:]]}
        elif UDOFactory.package_resolver:
            package_name_dict = UDOFactory.package_resolver
        else:
            package_name_dict = dict()

        # if there is no package lib provided, then it is assumed that the default qti package will be
        # will used to quantize any custom ops.
        if self.op_package_lib:
            self.quantize_with_default_package = False

        self.package_name_to_qnn_op_types = package_name_dict

    # overrides the resolve_package_names method in qnn_backend_base
    # to correctly resolve the package names for snpe 2.0 udo
    def resolve_package_names(self, node_type):
        default_package_name = qnn_definitions.QNN_OP_PACKAGE_NAME_QTI_AISW
        package_names = [default_package_name]
        for node_types, package_name in self.package_name_to_qnn_op_types.items():
            if node_type in [node_types]:
                package_names.append(package_name)
        return package_names[-1]

    def save(self, graph):
        self.dlc_serializer.initialize()
        log_info(code_to_message.get_progress_message("INFO_INITIALIZATION_SUCCESS"))
        # set up the package information for each op type in the graph
        self.set_package_dict(graph)
        # TODO: pass graph as-is
        ir_graph = self.get_ir_graph(graph)
        self.dlc_serializer.serialize(ir_graph)
        log_info(code_to_message.get_progress_message("INFO_CONVERSION_SUCCESS"))
        self.dlc_serializer.finish()
        log_info(code_to_message.get_progress_message("INFO_WRITE_SUCCESS"))
