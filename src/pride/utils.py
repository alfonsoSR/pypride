from functools import reduce


def factors(n: int) -> list[int]:
    """Factorize a number

    :return: List of factors
    """
    facs = set(
        reduce(
            list.__add__,
            [[i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0],
        )
    )
    return sorted(list(facs))
