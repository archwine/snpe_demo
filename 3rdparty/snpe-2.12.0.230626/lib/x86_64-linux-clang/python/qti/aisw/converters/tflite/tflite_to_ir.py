# ==============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.relay.relay_to_ir import RelayConverterFrontend
from qti.aisw.converters.relay.importers.tflite_importer import TFLiteImporter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
import numpy as np

class TFLiteConverterFrontend(RelayConverterFrontend):
    class ArgParser(RelayConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(TFLiteConverterFrontend.ArgParser, self).__init__(conflict_handler='resolve',
                                                                    parents=[TFLiteImporter.ArgParser()],
                                                                    **kwargs)

    def __init__(self, args, **kwargs):
        super(TFLiteConverterFrontend, self).__init__(args,
                                                      importer=TFLiteImporter(args),
                                                      axis_order=AxisOrders.TF,
                                                      **kwargs)
        # for pre-quantized model, the input dtypes may change after dequantize pass
        # we need to rewrite them
        for input_name, dtype in self.importer.dtype_dict.items():
            self.graph.input_dtypes_dict[input_name] = np.dtype(dtype)
