import time

def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic
