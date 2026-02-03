import logging
import os


def init_logger(log_file: str, name='acies'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    fmt = logging.Formatter(
        '%(levelname)-.1s%(asctime)s.%(msecs)06d %(thread)d %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y%m%d %I:%M:%S',
    )

    # console handler
    ch = logging.StreamHandler()
    level = os.environ.get('LOGLEVEL', 'WARN').upper()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    else:
        # update file handler
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        logger.addHandler(fh)

    return logger
