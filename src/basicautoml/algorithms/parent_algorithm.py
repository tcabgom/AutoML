

class ParentAlgorithm:

    def __init__(self):
        self.model = None
        self.params = None

    def get_name(self) -> str:
        raise NotImplementedError

    def get_algorithm_class(self) -> type:
        raise NotImplementedError

    def get_algorithm_params(self) -> dict:
        raise NotImplementedError
