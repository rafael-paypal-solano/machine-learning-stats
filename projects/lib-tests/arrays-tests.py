from environment import Arrays
import numpy

if __name__ == "__main__":
    a = Arrays.create(
        (
            (1,   2.0),
            (3.0, 1.0)
        )
    )
    b = numpy.array(((1,2.0),(3.0, 1)))
    print(Arrays.sum(a))
    print(Arrays.sum(b))

    A = Arrays.sum(a,0)
    B = Arrays.sum(b,0)

    print(type(A))

    print(Arrays.sum(a,1))
    print(Arrays.sum(b,1))        