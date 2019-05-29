"""
Runs some images through a pre-trained Alexnet model
"""

import tensorflow as tf
import numpy as np
from models import alexnet

# Constants
LOAD_PATH = "checkpoints/model.ckpt-115000"


def main():
    # create image tensor
    images = np.random.random((100, 224, 224, 3))
    image_tensor = tf.convert_to_tensor(images, dtype=tf.float32)

    # initialize model
    convnet = alexnet(image_tensor)

    # define output tensors of interest
    conv3_outputs = convnet.layers['conv3']

    # initialize tf Session and restore weighs
    sess = tf.Session()
    tf_saver_restore = tf.train.Saver()
    tf_saver_restore.restore(sess, LOAD_PATH)

    # run whatever tensors we care about
    conv3_outputs = sess.run(conv3_outputs)
    print(conv3_outputs.shape)

if __name__ == "__main__":
    main()
