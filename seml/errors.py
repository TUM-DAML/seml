import sys


class ConfigError(Exception):
    """Raised when the something is wrong in the config"""
    def __init__(self, message="The config file contains an error."):
        sys.tracebacklimit = 0
        super().__init__(message)


class ExecutableError(Exception):
    """Raised when the something is wrong with the Python executable"""
    def __init__(self, message="The Python executable has a problem."):
        sys.tracebacklimit = 0
        super().__init__(message)


class MongoDBError(Exception):
    """Raised when the something is wrong with the MongoDB"""
    def __init__(self, message="The MongoDB or its config has a problem."):
        sys.tracebacklimit = 0
        super().__init__(message)


class ArgumentError(Exception):
    """Raised when the something is wrong with the parsed arguments"""
    def __init__(self, message="The parsed arguments contain an error."):
        sys.tracebacklimit = 0
        super().__init__(message)
