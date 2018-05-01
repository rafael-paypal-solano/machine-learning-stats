

class ConvergenceState(object):
    def__init__(self, alpha=5-e6)
        self.__alpha__ = alpha
        self.__n__ = 0
        self.__in_range__ = False
        self.__in_range_count__ = 0
        self.__diff__ = 0.0

    @property
    def alpha(self):
        return self.__alpha__

    @property
    def n(self):
        return self.__n__

    def in_range(self):
        return self.__in_range__

    def __call__(self, diff):
        abs_diff = abs(diff)
        self.__in_range__ = if abs_diff < self.alpha
        self.__in_range_count__ += 1 if self.__in_range__ else 0
        self.__n__ += 1
        return self