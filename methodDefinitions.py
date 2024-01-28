import math


def protectedAdd(left, right):
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

# def avg(x):
#     return round(x, 0)
