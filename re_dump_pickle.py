import sys
import numpy as np

src = sys.argv[1]

byte_data = open('cluster/user_dict','rb')
data = open(src).read().replace('\r\n','\n')
dst = src + '.tmp'
open(dst,'w').write(data)

world = pickle.load(open(dst,'rb'),encoding='latin1')