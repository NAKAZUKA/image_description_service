class AppBaseException(Exception):
    status_code: int = 500

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class ModelLoadError(AppBaseException):
    pass


class TranslationError(AppBaseException):
    pass
