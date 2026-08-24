"""Logging utilities for the Mind project"""

import inspect
import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger in the 'mind' namespace

    Args:
        name: Optional logger name. If not provided, uses the calling module's __name__

    Returns:
        Logger instance in the mind namespace
    """
    if name is None:
        # Get caller's module name automatically
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'mind')

    # Ensure it's in the mind namespace
    if not name.startswith('mind'):
        if name == '__main__':
            name = 'mind'
        else:
            name = f'mind.{name.removeprefix("mind.")}'

    return logging.getLogger(name)