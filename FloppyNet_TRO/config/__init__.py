import os
import logging

## WORKSTATION-DEPENDANT PARAMETERS ###################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log_lvl = logging.INFO


def get_logger(log_lvl):
    

    FORMAT = '%(message)s'
    logging.basicConfig(format = FORMAT, level=logging.INFO)
    global_logger = logging.getLogger('global_logger')
    global_logger.setLevel(level=log_lvl)
     
    return global_logger


global_logger = get_logger(log_lvl)
