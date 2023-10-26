import os
import logging
import logging.config


def get_logger(name, level=None):
    from configs import config

    LOG_DIR = config['LOG_DIR']

    LOG_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '<%(filename)s-%(funcName)s()>: %(asctime)s, %(levelname)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'},
        },
        'handlers': {
            'default': {
                'level': level or 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': os.path.join(LOG_DIR, '%s.log' % name),
                'mode': 'a',
                'maxBytes': 100 * 1024 * 1024,  # 10 M
                # 'maxBytes': 10*1024,  # 10 M
                'backupCount': 3
            },
        },
        'root': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        }
    }

    '''
    handler = logging.FileHandler(os.path.join(folder, name+'.log'))
    formatter = logging.Formatter(
        '<%(filename)s-%(funcName)s()>: %(asctime)s,%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    if level is None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(level)
    logger.addHandler(handler)
    '''
    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(__name__)

    return logger
