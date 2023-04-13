import sys

from datetime import datetime


class FileLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


class GRLogger:
    def __init__(self):
        self.prefix = 'time'

    def info(self, s):
        print(f"[INFO] {str(datetime.now().strftime('<%H:%M:%S>'))}:\n {s}")

    def warning(self, s):
        print(f"[WARNING] {str(datetime.now().strftime('<%H:%M:%S>'))}:\n {s}")

    def error(self, s):
        print(f"[ERROR] {str(datetime.now().strftime('<%H:%M:%S>'))}:\n {s}")


if __name__ == '__main__':
    logger = GRLogger()
    logger.info('Hello FederatedScope!')
    logger.warning('Hello FederatedScope!')
    logger.error('Hello FederatedScope!')
