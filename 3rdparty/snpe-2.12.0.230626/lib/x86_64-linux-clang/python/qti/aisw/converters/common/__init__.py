# ==============================================================================
#
#  Copyright (c) 2022,2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys

if sys.version_info[0] == 3 and sys.version_info[1] == 6:
    from . import libPyIrGraph36 as ir_graph
else:
    from . import libPyIrGraph as ir_graph
