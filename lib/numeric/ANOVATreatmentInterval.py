class ANOVATreatmentInterval:
    def __init__(self, mean, length, pair):
        self.__mean__ = mean
        self.__length__ = length
        self.__pair__ = pair

    @property
    def mean(self):
        return self.__mean__

    @property
    def length(self):
        return self.__length__
    
    @property
    def pair(self):
        return self.__pair__