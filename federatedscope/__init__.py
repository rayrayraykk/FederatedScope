from __future__ import absolute_import, division, print_function

__version__ = '0.2.1'


def _setup_logger():
    """
    Setup the logger for the library.

    Args:
    """
    import logging

    logging_fmt = "%(asctime)s (%(module)s:%(lineno)d)" \
                  "%(levelname)s: %(message)s"
    logger = logging.getLogger("federatedscope")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging_fmt))
    logger.addHandler(handler)
    logger.propagate = False


_setup_logger()
