import sys
sys.path.append('./python/')
sys.path.append('./python/caffe/proto/')
from operator import itemgetter
import os
import re
from caffe.proto import caffe_pb2
import caffe.convert
from google.protobuf import text_format
import cPickle as pickle
import numpy as np

class CudaConvNetReader(object):
    def __init__(self, net, readblobs=False, ignore_data_and_loss=True):
        self.name = os.path.basename(net)
        self.readblobs = readblobs
        self.ignore_data_and_loss = ignore_data_and_loss

        try:
            net = pickle.load(open(net))
        except ImportError:
            # It wants the 'options' module from cuda-convnet
            # so we fake it by creating an object whose every member
            # is a class that does nothing
            faker = type('fake', (), {'__getattr__':
                                        lambda s, n: type(n, (), {})})()
            sys.modules['options'] = faker
            sys.modules['python_util.options'] = faker
            net = pickle.load(open(net))

        # Support either the full pickled net state
        # or just the layer list
        if isinstance(net, dict) and 'model_state' in net:
            self.net = net['model_state']['layers']
        elif isinstance(net, list):
            self.net = net
        else:
            raise Exception("Unknown cuda-convnet net type")

    neurontypemap = {'relu': caffe_pb2.V1LayerParameter.RELU,
                     'logistic': caffe_pb2.V1LayerParameter.SIGMOID,
                     'tanh' : caffe_pb2.V1LayerParameter.TANH}

    poolmethod = {
        'max': caffe_pb2.PoolingParameter.MAX,
        'avg': caffe_pb2.PoolingParameter.AVE 
    }

    def read(self):
        """
        Read the cuda-convnet file and convert it to a dict that has the
        same structure as a caffe protobuf
        """
        layers = []
        datalayer_vec = []

        def find_non_neuron_ancestors(layer):
            """Find the upstream layers that are not neurons"""
            out = []
            for l in layer.get('inputLayers', []):
                if l['type'] == 'neuron':
                    out += find_non_neuron_ancestors(l)
                else:
                    out += [l['name']]
            return out

        #for layer_key,layer in self.net.iteritems():
            #print layer_key

        for layer in self.net:

            layertype = layer['type'].split('.')[0]

            if (layertype == 'eltsum'):
                continue

            m = re.match('^data.*', layer['name'])
            if m != None:
                datalayer_vec.append( layer )

            if self.ignore_data_and_loss and layertype in ['data', 'cost']:
                continue

            readfn = getattr(self, 'read_' + layertype)

            convertedlayer = readfn(layer)

            # Add the top (our output) and bottom (input) links. Neuron layers
            # operate "in place" so have the same top and bottom.
            convertedlayer['bottom'] = find_non_neuron_ancestors(layer)

            if layer['type'] == "neuron" or layer['type'] == "relu":
                convertedlayer['top'] = convertedlayer['bottom']
            else:
                convertedlayer['top'] = [layer['name']]
            
            # prepare concat layer
            if ( len(convertedlayer['bottom']) >= 2 ):
                concat_name = '_'.join(convertedlayer['bottom'])
                concat_layer = {'type': caffe_pb2.V1LayerParameter.CONCAT, 'name': concat_name}
                concat_layer['top'] = [concat_name]
                concat_layer['bottom'] = convertedlayer['bottom']
                layers.append(concat_layer)
                convertedlayer['bottom'] = [concat_name]

            #print layerconnection
            layers.append(convertedlayer)

        saved_layers = layers
        layers = []

        known_names = ["data"]
        num_layers = len(saved_layers)
        neuron_layer_indice = []
        first_layer_idx = -1
        name2idx = {}
        name2neuron = {}
        for ldx in xrange(num_layers):
            if saved_layers[ldx]['name'] == 'fcf' and first_layer_idx == -1:
                first_layer_idx = ldx
            elif saved_layers[ldx]['type'] == caffe_pb2.V1LayerParameter.SOFTMAX:
                first_layer_idx = ldx
            elif saved_layers[ldx]['top'] == saved_layers[ldx]['bottom']:
                neuron_layer_indice.append(ldx)
                for n in saved_layers[ldx]['top']:
                    name2neuron[n] = ldx
            name2idx[saved_layers[ldx]['name']] = ldx

        depth = 0
        next_indice = [[first_layer_idx, depth]]
        saved_tuples = []
        
        while len(next_indice) > 0:

            top_tuple = next_indice[0]
            first_layer_idx,depth = top_tuple
            next_indice.remove(top_tuple)
            b_name = saved_layers[first_layer_idx]['top'][0]
            if name2neuron.has_key(b_name):
                saved_tuples.append([name2neuron[b_name], depth-1])
            saved_tuples.append([first_layer_idx,depth])
            #layers.append(saved_layers[first_layer_idx])

            print "layer {0}".format(len(layers))
            print "{0} - {1} : {2}".format(first_layer_idx, saved_layers[first_layer_idx]['name'], saved_layers[first_layer_idx]['bottom'])
            for b_name in saved_layers[first_layer_idx]['bottom']:
                if b_name in known_names:
                    break
                b_ldx = name2idx[b_name]
                if name2neuron.has_key(b_name):
                    #n_ldx = saved_layers[name2neuron[b_name]]
                    saved_tuples.append([name2neuron[b_name], depth+1])
                    next_indice.append([b_ldx, depth+2])
                else:
                    next_indice.append([b_ldx, depth+1])

        # sort by depth
        sorted_layer = sorted(saved_tuples, key=itemgetter(1), reverse=True)
        existed_ids = []
        for t in sorted_layer:
            if not t[0] in existed_ids:
                layers.append(saved_layers[t[0]])
                existed_ids.append(t[0])

        ## re-arrange the layers

        netdict = {'name': self.name, 
                   'layers': layers}

        # Add the hardcoded data dimensions instead of a data layer
        # will assume that the data layer is called "data" (since otherwise)
        # it is not trivial to distinguish it from a label layer)
        if self.ignore_data_and_loss != None and len(datalayer_vec) > 0:
            netdict['input'] = ["data"]
            size = int(np.sqrt(datalayer_vec[0]['outputs']/3))
            netdict['input_dim'] = [len(datalayer_vec), 3, size, size]
            #netdict['input_dim'] = [1, 3, size, size]
            ## add split layer
            if len(datalayer_vec) > 1:
                split_layer = {'type': caffe_pb2.LayerParameter.SLICE, 'name': 'data-slice', 'slice_param' : {'slice_dim' : 0}}
                split_layer['top'] = [datalayer['name'] for datalayer in datalayer_vec]
                split_layer['bottom'] = ['data']
                netdict['layers'].insert(0, split_layer)

        return netdict

    def read_data(self, layer):
        return {'type': 'data',
                'name': layer['name']
                }

    def read_conv(self, layer):
        assert len(layer['groups']) == 1
        assert layer['filters'] % layer['groups'][0] == 0
        assert layer['sharedBiases'] == True

        newlayer = {'type': caffe_pb2.V1LayerParameter.CONVOLUTION,
                    'name': layer['name'],
                    'convolution_param' :
                        {
                            'num_output': layer['filters'],
                            'weight_filler': {'type': 'gaussian',
                                              'std': layer['initW'][0]},
                            'bias_filler': {'type': 'constant',
                                            'value': layer['initB']},
                            'pad': -layer['padding'][0],
                            'kernel_size': layer['filterSize'][0],
                            'group': layer['groups'][0],
                            'stride': layer['stride'][0],
                        }
                    }

        if self.readblobs:
            # shape is ((channels/group)*filterSize*filterSize, nfilters)
            # want (nfilters, channels/group, height, width)

            print layer['weights'][0].shape
            
            weights = layer['weights'][0].T
            weights = weights.reshape(layer['filters'],
                                      layer['channels'][0]/layer['groups'][0],
                                      layer['filterSize'][0],
                                      layer['filterSize'][0])

            biases = layer['biases'].flatten()
            biases = biases.reshape(1, 1, 1, len(biases))

            weightsblob = caffe.convert.array_to_blobproto(weights)
            biasesblob = caffe.convert.array_to_blobproto(biases)
            newlayer['blobs'] = [weightsblob, biasesblob]

        return newlayer

    def read_local(self, layer):

        assert len(layer['groups']) == 1
        assert layer['filters'] % layer['groups'][0] == 0

        newlayer = {'type': caffe_pb2.V1LayerParameter.LOCAL,
                    'name': layer['name'],
                    'local_param' :
                        {
                            'num_output': layer['filters'],
                            'weight_filler': {'type': 'gaussian',
                                              'std': layer['initW'][0]},
                            'bias_filler': {'type': 'constant',
                                            'value': layer['initB']},
                            'pad': -layer['padding'][0],
                            'kernel_size': layer['filterSize'][0],
                            'bias_term' : True,
                            'stride': layer['stride'][0],
                        }
                    }

        if self.readblobs:

            # shape is ((channels/group)*filterSize*filterSize, nfilters)
            # want (nfilters, channels/group, height, width)

            weights = layer['weights'][0].T
            M = layer['filters']
            K = layer['channels'][0]/layer['groups'][0]*layer['filterSize'][0]*layer['filterSize'][0]
            N = layer['modules']
            for k in layer.keys():
                if k != 'inputLayers':
                    print "Key: {0}".format(k)
                    print layer[k]
            #print weights.shape
            #print layer['filters']
            #print layer['filterSize']
            #print layer['channels']
            #print layer['outputs']
            weights = weights.reshape(M, 1, K, N)

            biases = layer['biases'].flatten()
            assert(len(biases) == M*N)
            biases = biases.reshape(1, 1, M, N)

            weightsblob = caffe.convert.array_to_blobproto(weights)
            biasesblob = caffe.convert.array_to_blobproto(biases)
            newlayer['blobs'] = [weightsblob, biasesblob]

        return newlayer

    def read_pool(self, layer):
        return {'type': caffe_pb2.V1LayerParameter.POOLING,
                'name': layer['name'],
                'pooling_param' : {
                    'pool': self.poolmethod[layer['pool']],
                    'kernel_size': layer['sizeX'],
                    'stride': layer['stride'],
                    }
                }

    def read_fc(self, layer):
        newlayer = {'type': caffe_pb2.V1LayerParameter.INNER_PRODUCT,
                    'name': layer['name'],
                    'inner_product_param' : {
                        'num_output': layer['outputs'],
                        'weight_filler': {'type': 'gaussian',
                                          'std': layer['initW'][0]},
                        'bias_filler': {'type': 'constant',
                                        'value': layer['initB']},
                        }
                    }
        if self.readblobs:
            # shape is (ninputs, noutputs)
            # want (1, 1, noutputs, ninputs)
            ## Note FC can have Multiple inputs
            num_inputs = len(layer['weights'])
            tt_dim = sum([layer['numInputs'][x] for x in xrange(num_inputs)])
            weights = np.concatenate([x for x in layer['weights']]).T
            weights = weights.reshape(1, 1, layer['outputs'], tt_dim)

            biases = layer['biases'].flatten()
            biases = biases.reshape(1, 1, 1, len(biases))

            weightsblob = caffe.convert.array_to_blobproto(weights)
            biasesblob = caffe.convert.array_to_blobproto(biases)

            newlayer['blobs'] = [weightsblob, biasesblob]

        return newlayer

    def read_softmax(self, layer):
        return {'type': caffe_pb2.V1LayerParameter.SOFTMAX,
                'name': layer['name']}

    def read_cost(self, layer):
        # TODO recognise when combined with softmax and
        # use softmax_loss instead
        if layer['type'] == "cost.logreg":
            return {'type': caffe_pb2.V1LayerParameter.MULTINOMIAL_LOGISTIC_LOSS,
                    'name': layer['name']}

    def read_neuron(self, layer):
        assert layer['neuron']['type'] in self.neurontypemap.keys()
        return {'name': layer['name'],
                'type': self.neurontypemap[layer['neuron']['type']]}

    def read_cmrnorm(self, layer):
        return {'name': layer['name'],
                'type': caffe_pb2.V1LayerParameter.LRN,
                'lrn_param' : {
                    'local_size': layer['size'],
                    # cuda-convnet sneakily divides by size when reading the
                    # net parameter file (layer.py:1041) so correct here
                    'alpha': layer['scale'] * layer['size'],
                    'beta': layer['pow']
                    }
                }

    def read_rnorm(self, layer):
        # return self.read_cmrnorm(layer)
        raise NotImplementedError('rnorm not implemented')

    def read_cnorm(self, layer):
        raise NotImplementedError('cnorm not implemented')


class CudaConvNetWriter(object):
    def __init__(self, net):
        pass

    def write_data(self, layer):
        pass

    def write_conv(self, layer):
        pass

    def write_pool(self, layer):
        pass

    def write_innerproduct(self, layer):
        pass

    def write_softmax_loss(self, layer):
        pass

    def write_softmax(self, layer):
        pass

    def write_multinomial_logistic_loss(self, layer):
        pass

    def write_relu(self, layer):
        pass

    def write_sigmoid(self, layer):
        pass

    def write_dropout(self, layer):
        pass

    def write_lrn(self, layer):
        pass

def cudaconv_to_prototxt(cudanet):
    """Convert the cuda-convnet layer definition to caffe prototxt.
    Takes the filename of a pickled cuda-convnet snapshot and returns
    a string.
    """
    netdict = CudaConvNetReader(cudanet, readblobs=False).read()
    print "----------"
    message = caffe_pb2.NetParameter()
    protobufnet = dict_to_protobuf(netdict, message)

    return text_format.MessageToString(protobufnet)

def cudaconv_to_proto(cudanet):
    """Convert a cuda-convnet pickled network (including weights)
    to a caffe protobuffer. Takes a filename of a pickled cuda-convnet
    net and returns a NetParameter protobuffer python object,
    which can then be serialized with the SerializeToString() method
    and written to a file.
    """
    netdict = CudaConvNetReader(cudanet, readblobs=True).read()
    protobufnet = dict_to_protobuf(netdict)

    return protobufnet

# adapted from https://github.com/davyzhang/dict-to-protobuf/
def list_to_protobuf(values, message):
    """parse list to protobuf message"""
    if values == []:
        pass
    elif isinstance(values[0], dict):
        #value needs to be further parsed
        for val in values:
            cmd = message.add()
            dict_to_protobuf(val, cmd)
    else:
        #value can be set
        message.extend(values)

def dict_to_protobuf(values, message=None):
    """convert dict to protobuf"""
    if message is None:
        message = caffe_pb2.NetParameter()
        #for k,v in values.iteritems():
            #print "key: {0}".format(k)
            #print "value: {0}".format(v)
    #print "message: {0}".format(type(message))

    for k, val in values.iteritems():
        #print "k: {0}".format(k)
        if isinstance(val, dict):
            #value needs to be further parsed
            dict_to_protobuf(val, getattr(message, k))
        elif isinstance(val, list):
            list_to_protobuf(val, getattr(message, k))
        else:
            #value can be set
            setattr(message, k, val)

    return message
