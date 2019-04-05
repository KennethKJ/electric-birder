import tensorflow as tf
from keras.applications.vgg16 import preprocess_input


def _parse_function(filename, label, size, num_classes):

    # uncomment line for debug printing images being input
    filename = tf.Print(filename, [filename, label], "Pic&Lbl: ")

    # Convert to TF string
    image_string = tf.read_file(filename)
    image_jpeg = tf.image.decode_jpeg(image_string, channels=3)
    image_cast32 = tf.cast(image_jpeg, tf.float32)
    image_fl = image_cast32 / 255
    image_resized = tf.image.resize_image_with_pad(image_fl, size[0], size[1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    label = tf.one_hot(label, depth=num_classes)
    label = tf.Print(label, [label], "Pic&Lbl: ")

    return image_resized, label


def train_preprocess(image, label, use_random_flip):

    if use_random_flip:
        image_flipped = tf.image.random_flip_left_right(image)

    image_bright = tf.image.random_brightness(image_flipped, max_delta=0.3)
    image_sat = tf.image.random_saturation(image_bright, lower=0.5, upper=1.5)

    # Make sure image values are within its limits
    image_min = tf.minimum(image_sat, 1)
    image_max = tf.maximum(image_min, 0)

    image_flip = tf.image.random_flip_left_right(image_max)

    image_255 = tf.cast(tf.round(image_flip * 255), tf.int32)
    image_final = preprocess_input(image_255)

    # image_cast = tf.cast(image_255, tf.float32)
    #
    # image_final = image_cast - 116.779

    return image_final, label


def input_fn(filenames, labels, params, is_training):

    num_samples = len(filenames)
    # print("Num filenames passed inti input fn: " + str(len(filenames)))

    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset of images and labels
    parse_fn = lambda f, l: _parse_function(f, l, params['image size'], params['num classes'])
    train_fn = lambda f, l: train_preprocess(f, l, params['use random flip'])

    filenames = tf.constant(filenames)
    labels = tf.constant(labels)

    if is_training:
        dataset = (
            tf.data.Dataset.from_tensor_slices((filenames, labels))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params['num parallel calls'])
            .map(train_fn, num_parallel_calls=params['num parallel calls'])
            .batch(params["batch size"])
            .prefetch(2)  # make sure you always have one batch ready to serve
            .repeat(15)
        )
    else:
        dataset = (
            tf.data.Dataset.from_tensor_slices((filenames, labels))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params['num parallel calls'])
            .map(train_fn, num_parallel_calls=params['num parallel calls'])
            .batch(params["batch size"])
            .prefetch(2)  # make sure you always have one batch ready to serve
            .repeat(1)
        )

    # Set up iterator
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    # Point images to name of Kera's model input layer
    features = {'input_1': images}

    return features, labels
