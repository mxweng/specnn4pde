import time
from functools import wraps

def func_timer(function):
    """
    This is a timer decorator. It calculates the execution time of the function.
    
    Args:
    -------
    function : callable
        The function to be timed.

    Returns
    -------
    function : callable
        The decorated function which will print its execution time when called.

    Example
    -------
    >>> @func_timer
    >>> def my_function(n):
    >>>     return sum(range(n))
    >>> my_function(1000000)
    """
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Running time of %s: %.3e seconds" % (function.__name__, t1-t0))
        return result
    return function_timer

class ptimer():
    """
    This is a timer class. It provides start and stop methods to calculate the elapsed time.
    
    Methods:
        start: Starts the timer.
        stop: Stops the timer, calculates and prints the elapsed time.
        add: Add the time of the two timers together.
    """
    def  __init__(self):
        self.begin = 0
        self.end = 0
        self.lasted = []
        self.prompt = 'Timer has not started yet!'
        self.unit = ['y ', 'm ', 'd ', 'h ', 'm ', 's ']
        self.cycle = [12, 30, 24, 60, 60]

    def __str__(self):
        return self.prompt

    __repr__ = __str__

    def __add__(self, other):
        prompt = 'Total running time is: '
        result = []
        for i in range(6):
            result.append(self.lasted[i] + other.lasted[i])
        for i in range(5):
            if result[5 - i] >= self.cycle[4 - i]:
                result[4 - i] += 1
                result[5 - i] -= self.cycle[4 - i]
        for i in range(6):
            if result[i]:
                prompt += (str(result[i]) + self.unit[i])
        print(prompt)

    # Start timing
    def start(self):
        self.lasted = []
        self.begin = time.localtime()
        self.prompt = 'Timing has not stopped yet!'
        print('Timing started!')

    # Stop timing
    def stop(self):
        if not self.begin:
            print('Timer has not started yet!')
        else:
            self.end = time.localtime()
            print('Timing ended!')
            self._calc()

    # Internal function to calculate running time
    def _calc(self):
        self.prompt = 'Total running time is: '
        for i in range(6):
            self.lasted.append(self.end[i] - self.begin[i])
        for i in range(5):
            if self.lasted[5 - i] < 0:
                self.lasted[4 - i] -= 1
                self.lasted[5 - i] += self.cycle[4 - i]
        for i in range(6):
            if self.lasted[i]:
                self.prompt += (str(self.lasted[i]) + self.unit[i])
        print(self.prompt)
        self.begin = 0