from colorama import Fore, Style

class Colors:
    SUCCESS = Fore.GREEN
    ERROR = Fore.RED
    WARNING = Fore.YELLOW
    INFO = Fore.CYAN

def colorize(text, color):
    return f"{color}{text}{Style.RESET_ALL}"

def log_success(message):
    print(colorize(message, Colors.SUCCESS))

def log_error(message):
    print(colorize(message, Colors.ERROR))

def log_warning(message):
    print(colorize(message, Colors.WARNING))

def log_info(message):
    print(colorize(message, Colors.INFO))
