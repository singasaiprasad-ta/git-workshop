# Custom python functions

def double_number(a):
"""
Returns the double of the given number
Parameters:
    a (int or float): The number to double.
Returns:
    int or float: The result of doubling the input number.
"""
    print(f'value after double_number(): {a+a}')
    return a+a


def square_number(a):
"""
Returns the square of the given number.
Parameters:
    a (int or float): The number to square.
Returns:
    int or float: The result of squaring the input number.
"""
    print(f'value after square_number(): {a*a}')
    return a*a
