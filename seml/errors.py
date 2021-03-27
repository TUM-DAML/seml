class ConfigError(SystemExit):
    """Raised when the something is wrong in the config"""
    def __init__(self, message="The config file contains an error."):
        super().__init__(f"CONFIG ERROR: {message}")


class ExecutableError(SystemExit):
    """Raised when the something is wrong with the Python executable"""
    def __init__(self, message="The Python executable has a problem."):
        super().__init__(f"EXECUTABLE ERROR: {message}")


class MongoDBError(SystemExit):
    """Raised when the something is wrong with the MongoDB"""
    def __init__(self, message="The MongoDB or its config has a problem."):
        super().__init__(f"MONGODB ERROR: {message}")


class ArgumentError(SystemExit):
    """Raised when the something is wrong with the parsed arguments"""
    def __init__(self, message="The parsed arguments contain an error."):
        super().__init__(f"ARGUMENT ERROR: {message}")
