# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os

def setup_logger(verbose, output_dir='.'):
    formatter = '%(asctime)s - %(lineno)d - %(levelname)s - %(message)s'
    formatter = logging.Formatter(formatter)
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.name = "stream_handler"
    stream_handler.setLevel(lvl) # stream handler log level set depending on verbose
    stream_handler.setFormatter(formatter)

    log_file = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(log_file):
        os.mknod(log_file)
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    file_handler.name = "file_handler"
    file_handler.setLevel(logging.DEBUG) # file handler log level set to DEBUG so all logs are written to file
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # set base logger level to DEBUG so that all logs are caught by handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def setup_console_logger(verbose, output_dir='.'):
    formatter = '%(levelname)s: %(message)s'
    formatter = logging.Formatter(formatter)
    lvl = logging.INFO
    if verbose:
        lvl = logging.DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(lvl)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger('LS')
    logger.setLevel(lvl)
    logger.addHandler(stream_handler)
    return logger

def get_logger_log_file_path(logger, handler_name=None):
    file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    if len(file_handlers) is 0:
        return ""
    elif len(file_handlers) is 1:
        return file_handlers[0].baseFilename
    else: # in case of multiple file handlers
        file_handler = next((h for h in file_handlers if h.name is handler_name), None)
        if file_handler is not None:
            return file_handler.baseFilename
        else:
            return ""
