def sequence(v, p, column):
    subsequence = []
    for i in range(v):
        subsequence += [i] * v**(p - column)
    return subsequence * v**(column - 1)