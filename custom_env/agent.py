class Agent:
    """[summary] TODO 

    Args:

    Attributes:

    """
    def __init__(self, sight_distance) -> None:
        self._sight_distance: int = sight_distance
        self.policy = None # TODO
        pass

    @property
    def sight_distance(self) -> int:
        return self._sight_distance
    
    # def __repr__(self):
        # return self.__class__.__name__