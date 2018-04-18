import pycuda.autoinit
import skcuda.misc


class Info:
    def __init__(self):        
        if pycuda.driver.Device.count() > 0:
            self.__gpu_present__ = True
            skcuda.misc.init()
        else:
            self.__gpu_present__ = False

    def gpu_present(self):
        return self.__gpu_present__

InfoSingleton = Info()