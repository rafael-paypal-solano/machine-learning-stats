import convergence

class GammaToExponentialSeq(object):
    def __init__(self, A, B, N=100):
        self.__A__ = A
        self.__B__ = B
        self.__N__ = N
        self.__n__ = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.__n__ > self.__N__:
            raise StopIteration
        else:
            n = self.__n__  + 1
            self.__n__ = n
            a = numpy.power(self.__A__, n)
            b = numpy.
            return lambda x: self.
            

if __name__ == "__main__":
    print(tuple(GammaToExponentialSeq(0,1,10)))