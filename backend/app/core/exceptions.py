class PredixaException(Exception):
    """
    Base exception for Predixa AI.
    """

    pass


class ModelNotFoundError(PredixaException):
    pass


class DatasetNotFoundError(PredixaException):
    pass


class TrainingError(PredixaException):
    pass