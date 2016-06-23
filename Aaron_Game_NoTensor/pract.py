import os

try:
    f = open('./Weights/P1Weights1.npy', 'w')
except:
    print('')

try:
    f = open(os.path.dirname(os.path.dirname(__file__)) + '/Weights/P1Weights1.npy')
except:
    print('')

f.close()