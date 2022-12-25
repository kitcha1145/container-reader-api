__all__ = ['container_reader']


def container_reader():
    import sys, os
    # print(os.path.dirname(os.path.realpath(__file__)), __file__, f'{os.path.dirname(os.path.realpath(__file__))}/alpr_portable_t1')
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/container_reader')
    from container_reader.crda import PortA
    return PortA()
