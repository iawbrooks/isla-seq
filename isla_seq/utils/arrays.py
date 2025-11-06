import numpy as np


def random_filter(
        n: int, 
        p: float, 
        rng: int | np.random.RandomState | None = None, 
        independent: bool = False,
    ) -> np.ndarray:
    """
    Generate a 1D boolean filter with a set probability of True values.

    Parameters
    ---
    n : `int`
        The size of the generated filter.
    p : `float`, between 0 and 1, inclusive
        The probability that any given value will be True.
    rng : `int` | `RandomState` | `None`, default: `None`
        An integer seed or RandomState object to use for generating the filter.
    independent: `bool`, default: `False`
        Whether each value's probability of being True should be independent of all the other
        values. When True, the number of total True values may vary, as each individual
        value's probability of being True will be equal to p regardless of the rest.
        When False, the number of True values in the generated array is predetermined
        to be `n * p`. 
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1, inclusive")
    
    # If predetermined number of True values, use random_filter_n
    if not independent:
        n_true = int(n * p)
        return random_filter_n(n, n_true, rng)
    
    # Otherwise, generate independently
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    return rng.random(n) < p


def random_filter_n(
        n: int, 
        n_true: int, 
        rng: int | np.random.RandomState | None = None
    ) -> np.ndarray:
    """
    Generate a 1D boolean filter with a set number of True values.

    Parameters
    ---
    n : `int`
        The size of the generated filter.
    n_true : `int`, between 0 and n, inclusive
        The number of values that will be True in the filter.
    rng : `int` | `RandomState` | `None`, default: `None`
        An integer seed or RandomState object to use for generating the filter.
    """
    if not 0 <= n_true <= n:
        raise ValueError("n_true must be between 0 and n, inclusive")
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    
    filt = np.zeros(n, dtype=bool)
    filt[ : n_true] = True
    rng.shuffle(filt)

    return filt
