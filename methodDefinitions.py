import math


def protected_add(left, right):
    try:
        result = left + right
        if math.isinf(result) or math.isnan(result):
            raise OverflowError("Result is infinity or NaN")
        return result
    except OverflowError as e:
        print(f"Warning: {e}")
        print("Left:", left)
        print("Right:", right)
        return 1


def sqrt(x):
    if x < 0:
        return 1
    return math.sqrt(x)


def sin(x):
    return math.sin(x)


def pow2(x):
    return x * x


def pow3(x):
    return x * x * x


def custom_and(x, y):
    return x and y


def custom_or(x, y):
    return x or y


def custom_not(x):
    return not x


def custom_if(x, y, z):
    if x:
        return y
    else:
        return z

# def avg(x):
#     return round(x, 0)
