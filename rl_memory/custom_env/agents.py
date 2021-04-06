class Agent:
    """[summary] TODO -> docs

    Args:

    Attributes:

    """
    def __init__(self, sight_distance: int) -> None:
        self._sight_distance: int = sight_distance 
        self.policy = None # TODO
        pass # TODO

    @property
    def sight_distance(self) -> int:
        return self._sight_distance

    @sight_distance.setter
    def sight_distance(self, value: int):
        if isinstance(value, int):
            self._sight_distance = value
        else:
            raise TypeError("'sight_distance' must be an integer.")
    
    # def __repr__(self):
        # return self.__class__.__name__

class ValueAgent(Agent):
    pass # TODO

class PolicyAgent(Agent):
    pass # TODO

class AdvantageAgent(Agent):
    pass # TODO