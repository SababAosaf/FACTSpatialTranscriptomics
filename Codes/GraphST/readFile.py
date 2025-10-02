import pickle

import numpy as np

with open('D:\Projects\DeepST\deepst\\filename.pickle', 'rb') as f:
    x = pickle.load(f)
with open('D:\Projects\conST\\filename.pickle', 'rb') as f:
    y = pickle.load(f)



#
# np.savetxt('test.out', x, delimiter=',',fmt='%1.2f')
# np.savetxt('test1.out', y, delimiter=',',fmt='%1.2f')
#
# print(x)
print(y.shape)

# with open('D:\Projects\conST\\filename.pickle', 'rb') as f:
#     x = pickle.load(f)
#     print(x)
#     print(x.shape)