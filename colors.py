"""
This module provides ANSI escape codes for colorizing console output.
These colors can be used to enhance the readability of console messages.
"""

from enum import Enum
from typing import NewType

ANSIColor = NewType('ANSIColor', str)

class Colors(Enum):
    PINK: ANSIColor = ANSIColor('\033[95m')
    CYAN: ANSIColor = ANSIColor('\033[96m')
    YELLOW: ANSIColor = ANSIColor('\033[93m')
    NEON_GREEN: ANSIColor = ANSIColor('\033[92m')
    RESET: ANSIColor = ANSIColor('\033[0m')

# For backwards compatibility, also provide the colors as top-level constants
PINK: ANSIColor = Colors.PINK.value
CYAN: ANSIColor = Colors.CYAN.value
YELLOW: ANSIColor = Colors.YELLOW.value
NEON_GREEN: ANSIColor = Colors.NEON_GREEN.value
RESET_COLOR: ANSIColor = Colors.RESET.value
