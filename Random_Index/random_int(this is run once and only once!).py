import pandas as pd
from random import shuffle


# #725 is the size of doctags from doc2vec

x = [i for i in range(1000)]

# we want 725*.2=145 out of this:
shuffle(x)

test_ind = x[:200]
train_ind = x[200:]

print(test_ind)
print(train_ind)

# test_ind = [68, 98, 715, 371, 535, 559, 117, 32, 412, 675, 486, 406, 146, \
# 83, 593, 115, 611, 73, 679, 151, 312, 639, 508, 685, 282, 599, 395, 177, \
# 657, 456, 455, 109, 621, 600, 606, 349, 475, 125, 620, 437, 277, 149, 301, \
# 297, 607, 90, 37, 492, 352, 560, 124, 44, 589, 85, 130, 515, 197, 188, 574, \
# 96, 5, 648, 104, 575, 481, 635, 262, 396, 239, 617, 8, 401, 103, 602, 598, 71, \
# 460, 688, 79, 36, 627, 487, 357, 2, 46, 61, 274, 359, 466, 200, 407, 415, 644, \
# 326, 142, 308, 91, 510, 507, 638, 193, 543, 152, 218, 280, 31, 670, 240, 145, 101, \
# 520, 304, 270, 321, 358, 370, 311, 33, 653, 182, 683, 708, 568, 582, 300, 289, 491, \
# 678, 363, 332, 427, 555, 154, 482, 333, 476, 105, 226, 645, 141, 258, 351, 102, 353, 155]

# train_ind = [i for i in range(725) if i not in test_ind]

# train_ind = pd.Series(train_ind)
# test_ind = pd.Series(test_ind)

# train_ind.to_pickle("train_ind.pkl")
# test_ind.to_pickle("test_ind.pkl")