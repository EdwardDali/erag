from colorama import Fore, Style

class Colors:
    INFO = Fore.CYAN
    SUCCESS = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    RESET = Style.RESET_ALL

def colorize(text, color):
    return f"{color}{text}{Colors.RESET}"
