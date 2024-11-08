import random
import string


def safe_div(x,y):
    if y == 0:
        return None
    return x / y

def random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))