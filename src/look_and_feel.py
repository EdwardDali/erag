# ANSI escape codes for text colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Custom colors
DUSTY_PINK = "\033[38;2;173;142;148m"
SAGE_GREEN = "\033[38;2;125;169;133m"

# ANSI escape codes for text styles
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# ANSI escape code to reset all formatting
RESET = "\033[0m"

def colorize(text, color, style=""):
    """
    Wrap the given text with color and style ANSI escape codes.
    """
    return f"{style}{color}{text}{RESET}"

# Predefined color functions for common use cases
def error(text):
    return colorize(text, RED, BOLD)

def success(text):
    return colorize(text, GREEN)

def warning(text):
    return colorize(text, YELLOW)

def info(text):
    return colorize(text, CYAN)

def highlight(text):
    return colorize(text, MAGENTA, BOLD)

# New color functions for user input and LLM response
def user_input(text):
    return colorize(text, DUSTY_PINK)

def llm_response(text):
    return colorize(text, SAGE_GREEN)
