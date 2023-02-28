class IteratorAccess(object):
    FORMAT = "base"

    def __init__(self, file_name: str, **kwargs) -> None:
        self.file_name = file_name

    def __iter__(self):
        return self

    def __next__(self):
        pass
