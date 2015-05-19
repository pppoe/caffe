#! /usr/bin/env python

import sys
sys.path.append('./python/caffe/')
sys.path.append('./python/caffe/proto/')
from convert_net import *

if len(sys.argv) < 2:
    print "usage:- CNNPath"
    sys.exit(-1)

CNNPath=sys.argv[1]
print CNNPath

netpt = cudaconv_to_prototxt(CNNPath)
fh=open(CNNPath+"-prototxt", 'w')
fh.write(netpt)
fh.close()

netpb = cudaconv_to_proto(CNNPath)
fh=open(CNNPath + "-proto", 'wb')
fh.write(netpb.SerializeToString())
fh.close()
