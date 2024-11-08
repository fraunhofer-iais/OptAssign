from typing import Callable, Optional


class ProblemSize(Exception):
    pass





class NoModelAvailable(Exception):
    pass


class FunctionOrderError(Exception):
    """Exception raised when functions are called in the wrong order."""

    def __init__(
        self,
        current_function: Callable,
        required_function: Callable,
        message: Optional[str] = None,
    ):
        if message is None:
            message = "Please call the required function first."
        self.current_function = current_function
        self.required_function = required_function
        self.message = f"'{current_function.__qualname__}' requires '{required_function.__qualname__}' to be called first.\n{message}"
        super().__init__(self.message)
