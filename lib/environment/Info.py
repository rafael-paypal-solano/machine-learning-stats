import pycuda.autoinit

class Info:

    @classmethod
    def gpu_present(clazz):
        return pycuda.driver.Device.count() > 0