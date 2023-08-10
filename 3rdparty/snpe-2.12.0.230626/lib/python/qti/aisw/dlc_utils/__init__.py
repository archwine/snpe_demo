# ==============================================================================
#
#  Copyright (c) 2019-2020,2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import sys

try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        from . import libDlModelToolsPy36 as modeltools
        from qti.aisw.converters.common import libPyIrGraph36 as ir_graph
        from . import libDlContainerPy36 as dlcontainer
    else:
        from . import libDlModelToolsPy3 as modeltools
        from qti.aisw.converters.common import libPyIrGraph as ir_graph
        from . import libDlContainerPy3 as dlcontainer
except ImportError as ie1:
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            import libDlModelToolsPy36 as modeltools
            import libPyIrGraph36 as ir_graph
            import libDlContainerPy36 as dlcontainer
        else:
            import libDlModelToolsPy3 as modeltools
            import libPyIrGraph as ir_graph
            import libDlContainerPy3 as dlcontainer
    except ImportError:
        raise ie1
