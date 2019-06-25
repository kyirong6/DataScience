import time
#from exercises.E_6 import implementations    ----had to use this import due to IDE and project structure-----
from implementations import all_implementations
import numpy as np
import pandas as pd



def main():

    #all_implementations = implementations.all_implementations    ----due to ide and project structure---

    q = 0
    data = pd.DataFrame()
    names = ["qs1", "qs2", "qs3", "qs4", "qs5", "merge1", "partition_sort"]

    for sort in all_implementations:
        val = np.empty(50)
        for i in range(0, 50):
            random_array = np.random.randint(100000, size=16000)
            st = time.time()
            res = sort(random_array)
            en = time.time()
            val[i] = en-st
        data[names[q]] = val
        q += 1

    data.to_csv('data.csv', index=False)


if __name__ == '__main__':
    main()

