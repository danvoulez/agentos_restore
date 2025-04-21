# promptos_backend/app/core/exceptions.py
class CustomException(Exception):
    """Custom exception class."""
    def __init__(self, message: str):
        super().__init__(message)
