# ==============================================================================
#
#  Copyright (c) 2021, 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.relay.relay_to_ir import RelayConverterFrontend
from qti.aisw.converters.relay.importers.pytorch_importer import PyTorchImporter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrders
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings
from qti.aisw.converters.common.utils.converter_utils import log_debug1


class PyTorchConverterFrontend(RelayConverterFrontend):
    class ArgParser(RelayConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(PyTorchConverterFrontend.ArgParser, self).__init__(conflict_handler='resolve',
                                                                    parents=[PyTorchImporter.ArgParser()],
                                                                    **kwargs)

    def __init__(self, args, **kwargs):
        super(PyTorchConverterFrontend, self).__init__(args,
                                                      importer=PyTorchImporter(args),
                                                      axis_order=AxisOrders.PYTORCH,
                                                      **kwargs)

        # set default input_layout if user doesn't specify it in command
        for input_name, input_shape in self.importer.shape_dict.items():
            if input_name not in self.graph.input_axis_formats:
                # handle time_series formats based on input enconding
                encodings = self.graph.get_input_encodings()
                input_in_encodings = [input_encoding[0] for input_encoding in encodings]
                if InputEncodings.OTHER in input_in_encodings and len(input_shape) in [3, 4]:
                    self.graph.input_axis_formats[input_name] = AxisTracker.AxisFormat.NONTRIVIAL
                else:
                    # Override time_series_format based on encoding
                    time_series_format = False
                    if InputEncodings.TIME_SERIES in input_in_encodings and len(input_shape) == 3:
                        time_series_format = True
                    self.graph.input_axis_formats[input_name] = self.graph.src_axis_order.get_default_input_axis_format(len(input_shape),
                                                                                                                        time_series_format=time_series_format)
            log_debug1("Set input axis-format for {} with {}".format(input_name, self.graph.input_axis_formats[input_name]))