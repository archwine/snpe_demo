# ==============================================================================
#
#  Copyright (c) 2019-2020,2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# @deprecated
# to allow for backward compatibility adding this import also at top-level so that
# it is possible to do <from qti.aisw import modeltools>
# moving forward the signature to use will be <from qti.aisw.dlc_utils import modeltools>
try:
    import sys
    from qti.aisw.converters.backend.ir_to_dlc import DLCBackend as NativeBackend
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        from qti.aisw.dlc_utils import libDlModelToolsPy36 as modeltools
        from qti.aisw.dlc_utils import libDlContainerPy36 as dlcontainer
    else:
        from qti.aisw.dlc_utils import libDlModelToolsPy3 as modeltools
        from qti.aisw.dlc_utils import libDlContainerPy3 as dlcontainer
except ImportError as ie:
    raise ie
