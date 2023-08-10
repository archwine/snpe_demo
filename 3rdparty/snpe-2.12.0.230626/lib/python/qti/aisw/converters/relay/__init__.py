# ==============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

# Make sure the TVM library is loaded from SDK
import os
import sys
qti_path = os.path.dirname(sys.modules['qti'].__file__)
os.environ["TVM_LIBRARY_PATH"] = os.path.join(qti_path, "tvm")

# TVM needs to be imported here first to introduce the module name "qti.tvm" replacement
# logic in python/tvm/__init__.py, so the following python files can directly use
# "import tvm" to import the TVM module
from qti import tvm