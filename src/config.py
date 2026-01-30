"""
Configuration module for NutriGen AI.

This module centralizes all configuration settings including:
    - Directory paths for data files
    - ANSI color codes for terminal UI
    - Shared constants across the application

All modules should import configuration from here to maintain consistency.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Dynamic path resolution - DATA_DIR points to the 'data' folder at project root
DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)


# =============================================================================
# TERMINAL STYLING
# =============================================================================


class Colors:
    """
    ANSI escape codes for terminal styling.

    Provides both raw color codes and semantic aliases for
    consistent UI feedback throughout the application.

    Attributes:
        HEADER: Purple/magenta for headers.
        BLUE: Blue for general highlights.
        CYAN: Cyan for informational messages.
        GREEN: Green for success messages.
        YELLOW: Yellow for warnings.
        RED: Red for errors.
        WHITE: White for regular text.
        BOLD: Bold text modifier.
        DIM: Dimmed text modifier.
        UNDERLINE: Underlined text modifier.
        END: Reset all formatting.
    """

    HEADER: str = "\033[95m"
    BLUE: str = "\033[94m"
    CYAN: str = "\033[96m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    RED: str = "\033[91m"
    WHITE: str = "\033[97m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"
    UNDERLINE: str = "\033[4m"
    END: str = "\033[0m"

    # Semantic aliases for consistent UI messaging
    SUCCESS: str = GREEN
    WARNING: str = YELLOW
    ERROR: str = RED
    INFO: str = CYAN
    ACCENT: str = BLUE
