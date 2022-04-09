import logging


def set_logger(log_file,log_level):
    pass
    file_handler = logging.FileHandler(filename=log_file,encoding="utf-8")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.setLevel(log_level)
    stream_handler.setLevel(log_level)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)
    return logger




