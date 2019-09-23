# -*- coding: utf-8 -*-
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import os
from google.protobuf import text_format
import matplotlib.pyplot as plt

class Model_net:

    def __init__(self, caffemodel, deply_file, mean_file=None, gpu=False, device_id=0):
        os.environ['GLOG_minloglevel'] = '2' #0-debug, 1-info 2-warnings 3-errors
        if gpu:
            caffe.set_device(device_id)
            caffe.set_mode_gpu()
            print("GPU mode")
        else:
            caffe.set_mode_cpu()
            print("CPU mode")
        self.net = caffe.Net(caffemodel, deply_file, caffe.TEST)
        self.transformer = self.get_transformer(deply_file, mean_file)

    def get_transformer(self, deploy_file, mean_file=None):
        # input_dim: 1
        # input_dim: 3
        # input_dim: 224
        # input_dim: 224
        network = caffe_pb2.NetParameter()
        with open(deploy_file) as infile:
            text_format.Merge(infile.read(), network)
        if network.input_shape:
            dims = network.input_shape[0].dim
        else:
            dims = network.input_dim[:4]
        t = caffe.io.Transformer(inputs={'data':dims})
        # Set the order of dimensions, e.g. to convert OpenCV's HxWxC images
        #         into CxHxW.
        t.set_transpose('data',(2,0,1)) # channel, H, W

        if dims[0] == 3:
            t.set_channel_swap('data',(2,1,0))

        if mean_file:
            with open(mean_file,'rb') as infile:
                # （Num，Channels，Height，Width) N K H W
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(infile.read())
                if blob.HasField('shape'):
                    blob_dims = blob.shape.dim
                    assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is %s' % blob.shape
                elif blob.HasField('num') and blob.HasField('channels') and blob.HasField('height') and blob.HasField(
                        'width'):
                    blob_dims = (blob.num, blob.channels, blob.height, blob.width)  # KNHW
                else:
                    raise ValueError('blob does not provide shape or 4d dimensions')

                pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1) # return the mean value of the given axis
        else:
            pixel = [129, 104, 93]
            t.set_mean('data', np.array(pixel))

    def forward_pass(self, images, transformer, batch_size=1, layer=None):
        caffe_images = []
        # K H W
        for image in images:
            # ndim number of dimensions
            if image.ndim == 2:
                caffe_images.append(image[:,:,np.newaxis])
            else:
                caffe_images.append(image)
        caffe_images = np.array(caffe_images)
        # transformer.inputs, a function in caffe
        dims = transformer.inputs['data'][1:] # channel, width, height

        scores = None
        fea = None

        for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
            new_shape = (len(chunk), ) + tuple(dims) # when % != 0
            if self.net.blobs['data'].data.shape != new_shape:
                self.net.blobs['data'].reshape(*new_shape)
            for idx, img in enumerate(chunk):
                image_data = transformer.preprocess('data', img)
                self.net.blobs['data'].data[idx] = image_data
            output = self.net.forward()[self.net.outputs[-1]]

            if layer is not None:
                if fea is None:
                    fea = np.copy(self.net.blobs[layer].data)
                else:
                    fea = np.vstack((fea, self.net.blobs[layer].data))

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))

    def classify(self, image_list, layer_name=None):
        _, channels, heght, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels ==1:
            mode = 'L'
        else:
            raise ValueError('Invalid channel number: %s' % channels)

        fea = None
        scores, fea = self.forward_pass(image_list, self.transformer, batch_size=1, layer=layer_name)
        return  (scores, np.argmax(scores,1), fea)

    def showimage(self, im):
        if im.ndim == 3:
            im = im[:, :, ::-1]
        plt.imshow(im)
        plt.show()

    def vis_square(self, data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:].transpose(0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        self.showimage(data)

if __name__ == '__main__':
    caffemodel = './deep_model/VGG_FACE.caffemodel'
    deploy_file = './deep_model/VGG_FACE_deploy.prototxt'
    mean_file = None

    gpu = True
    net = Model_net(caffemodel, deploy_file, mean_file, gpu)