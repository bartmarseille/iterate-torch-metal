import numbers

def format_num(num, n_dec=3, fmt='g'):
    c = 'e' if num > 1e4 else fmt
    n_dec += len(str(int(num))) if c=='g' else 0
    return '{num:.{dec}{c}}'.format(num=num, dec=n_dec, c=c)


def to_str(P, n_dec=3, fmt='g'):
    p_str = 'undecided'
    if isinstance(P, (bool, str)):
        p_str = f"'{P}'"
    elif isinstance(P, numbers.Number):
        p_str = format_num(P, n_dec, fmt)
    elif hasattr(P, "__len__"):
        p_str = '[' + ', '.join([to_str(p, n_dec, fmt) for p in P]) + ']'
    else:
        p_str = f"'{P}'"
    return p_str