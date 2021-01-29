import logging

def setup_logging(log_level=logging.INFO):
    """
    Common logging mechanism - call at beginning of application
    Returns:
    """
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)