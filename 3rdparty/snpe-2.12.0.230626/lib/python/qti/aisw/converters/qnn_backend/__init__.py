# ==============================================================================
#
#  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        from qti.aisw.converters.common import libPyIrGraph36 as ir_graph
        from . import libPyQnnDefinitions36 as qnn_definitions
        from qti.aisw.converters.common import libPyIrSerializer36 as qnn_ir
    else:
        from qti.aisw.converters.common import libPyIrGraph as ir_graph
        from . import libPyQnnDefinitions as qnn_definitions
        from qti.aisw.converters.common import libPyIrSerializer as qnn_ir

except ImportError as e:
    try:
        import libPyQnnDefines as qnn_definitions
    except ImportError:
        raise ImportError(e)
