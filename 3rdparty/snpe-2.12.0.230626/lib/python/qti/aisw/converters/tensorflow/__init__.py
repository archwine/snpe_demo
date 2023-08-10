# =============================================================================
#
#  Copyright (c) 2021,2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except AttributeError:
    tf_compat_v1 = tf

    # import contrib ops since they are not imported as part of TF by default
    import tensorflow.contrib


# Do some quick validation of python and tensorflow versions
tf_ver = tf.__version__.split('.')

# If python 3.8 then must be TF 2.x
if sys.version_info[0] == 3 and sys.version_info[1] == 8:
    if int(tf_ver[0]) != 2:
        raise ValueError("Only Tensorflow 2.x supported with python 3.8")
# If python 3.6 then must be TF 1.15
elif sys.version_info[0] == 3 and sys.version_info[1] == 6:
    if int(tf_ver[0]) != 1 and int(tf_ver[1]) != 15:
        raise ValueError("Only Tensorflow 1.15 supported with python 3.6")
else:
    raise ValueError("Unsupported python {} and tensorflow {} version combination".format(sys.version, tf.__version__))
