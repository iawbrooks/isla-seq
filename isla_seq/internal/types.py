
def is_iterable(v):
    try:
        iter(v)
        return True
    except TypeError:
        return False
