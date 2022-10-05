import torch
from torch import multiprocessing as mp

buffers = []
n = 0
while True:
  try:
    n += 1
    buffers.append(torch.zeros((10)).share_memory_())
    if n % 10000 == 0:
      print(n)
  except Exception as e:
    print("max buffers", n)
    raise e