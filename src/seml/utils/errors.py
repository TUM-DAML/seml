class InputError(SystemExit):
    """Parent class for input errors that don't print a stack trace."""

    pass


class ConfigError(InputError):
    """Raised when the something is wrong in the config"""

    def __init__(self, message='The config file contains an error.'):
        super().__init__(f'CONFIG ERROR: {message}')


class ExecutableError(InputError):
    """Raised when the something is wrong with the Python executable"""

    def __init__(self, message='The Python executable has a problem.'):
        super().__init__(f'EXECUTABLE ERROR: {message}')


class MongoDBError(InputError):
    """Raised when the something is wrong with the MongoDB"""

    def __init__(self, message='The MongoDB or its config has a problem.'):
        super().__init__(f'MONGODB ERROR: {message}')


class ArgumentError(InputError):
    """Raised when the something is wrong with the parsed arguments"""

    def __init__(self, message='The parsed arguments contain an error.'):
        super().__init__(f'ARGUMENT ERROR: {message}')
