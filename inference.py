
def binary_classification_from_raw_images(model, weights):
    #### set your paths
    # my_model_mean_binaryproto = my trained images mean binaryproto.
    # my_model_data = path to my images folder. all my images are in the same folder and i differentiate them by keywords in the filename.
    #                 In my images folder I also have val.txt and train.txt with filename and their label for training and validation.
    #                 From these files I also create the lmdbs.

    # * Load `caffe`.
    # The caffe module needs to be on the Python path;
    #  we'll add it here explicitly.


    # * Set Caffe to CPU mode and load the net from disk.
    caffe.set_mode_gpu()

    net = caffe.Net(model,weights,caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this, but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
    #
    #     Our default CaffeNet is configured to take images in BGR format. Values are expected to start in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them. In addition, the channel dimension is expected as the first (_outermost_) dimension.
    #
    #     As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the _innermost_ dimension, we are arranging for the needed transformations here.

    #### I don't have .npy array with the mean.
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    # mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    # mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    # print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    # I don't have my image mean in .npy file but in binaryproto. I'm converting it to a numpy array.
    # Took me some time to figure this out.
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open('/home/monete/monete@gmail.com/coding/MAI/thesis/inference/mean.binaryproto', 'rb').read()
    blob.ParseFromString(data)
    mu = np.array(caffe.io.blobproto_to_array(blob))
    mu = mu.squeeze() # The output array had one redundant dimension.

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 224)  # rescale from [0, 1] to [0, 255] - needed for caffenet\Alexnet.
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # ### 3. CPU classification
    #
    # * Now we're ready to perform classification. Even though we'll only classify one image, we'll set a batch size of 50 to demonstrate batching.

    # In[6]:

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(1,  # batch size
                              3,  # 3-channel (BGR) images
                              224, 224)  # image size is 227x227

    correct = 0
    count = 1

    # save labels and prediction in arrays for further statistical analysis.
    y_test, y_pred = [], []

    # in my case, I'm reading the images file names from val.txt.

    val_images = open(my_model_data + "/val3.txt").readlines()

    # You don't need to shuffle.. This is how I want to see the output.
    random.shuffle(val_images)

    # I'm saving mis-classified filenames in a file.
    misclassified = open(my_model_data + "/misclassified.txt", "w")

    for image_name_n_label in val_images:
        #in val.txt every line contains - filename label
        #image_file, label = my_model_data + "/" + image_name_n_label.split(' ')[0], int(image_name_n_label.split(' ')[1])
        image_file, label = image_name_n_label.split(' ')[0], int(image_name_n_label.split(' ')[1])
        y_test.append(label)

        print(count, ". image: " + os.path.basename(image_file) + " label: " + str(label))
        image = caffe.io.load_image(image_file)
        print (image.shape[0])
        # image shape is (3, 256, 256). we want it (3, 227, 227) for caffenet.
        if image.shape[0] != CLASSIFICATION_IMAGE_SIZE:

            # I'm cropping the numpy array on the fly so that I don't have to mess with resizing
            # the actual images in a separate folder each time.
            image = center_crop_image(image, CLASSIFICATION_IMAGE_SIZE, CLASSIFICATION_IMAGE_SIZE)
            if image.shape[0] != CLASSIFICATION_IMAGE_SIZE:
                print("!!!!!!! cropped shape ", image.shape)

        transformed_image = transformer.preprocess('data', image)


        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()

        output_prob = output['Softmax'][0]  # the output probability vector for the first image in the batch
        print(count, ". prob for neutral class: {0:.5f}, prob for anger class: {1:.5f} prob for disgust class: {2:.5f}, prob for fear class: {3:.5f}, prob for happy class: {4:.5f}, prob for sadness class: {5:.5f}, prob for surpise class: {6:.5f}".format(output_prob[0], output_prob[1], output_prob[2], output_prob[3], output_prob[4], output_prob[5], output_prob[6]))
        predicted_label = output_prob.argmax()
        y_pred.append(predicted_label)



        if predicted_label == label:
            correct += 1
        else:
            print("!!!!!!!!!!!!!!! misclassified")
            misclassified.write(os.path.basename(image_file) + " " + str(label) + " " + str(predicted_label) + "\n")
            # display misclassified images.
            #image = PIL.Image.open(image_file)
            #image.show()



        accuracy = ((100. * correct) / (count))


        print(count, '. predicted class is: ', output_prob.argmax())
        print(count, ". accuracy: " + str(accuracy))

        print("")

        count += 1

    misclassified.close()
# -------------------------------------------------------------------------------------------------------

def center_crop_image(image, new_width, new_height):

    height, width, chan = image.shape

    width_cut = (width - new_width) // 2
    height_cut = (height - new_height) // 2

    top, bottom = height_cut, -height_cut
    left, right = width_cut, -width_cut

    # could have 1 pixel off.
    height_diff = new_height - (height - (height_cut*2))
    width_diff = new_width - (width - (width_cut*2))

    top -= height_diff
    left -= width_diff

    # or
    # bottom += ydiff
    # right += xdiff
    # or any other combination

    return image[top:bottom, left:right]

# -------------------------------------------------------------------------------------------------------

import random
import os
import PIL
import caffe
import sys
import numpy as np

caffe_root = '/home/monete/monete@gmail.com/coding/caffe'
sys.path.insert(0, caffe_root + '/python')
model = '/home/monete/monete@gmail.com/coding/MAI/thesis/inference/resnet/deploy.prototxt'
weights = '/home/monete/monete@gmail.com/coding/MAI/thesis/inference/resnet/resnet.caffemodel'
CLASSIFICATION_IMAGE_SIZE=224
my_model_data = '/home/monete/monete@gmail.com/coding/MAI/thesis/inference/'
binary_classification_from_raw_images(model, weights)
