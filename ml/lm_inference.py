import os
import pickle
import logging

import numpy as np

from PIL import Image
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAveragePooling2D, concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

IMAGE_SIZE = (None, None, 3)
N_CLASSES = 29
# change them
PREDICT_IMAGES_ROOT = '/home/andreyzharkov/data/data_29_full'
PREDICT_LOG_DIR = './logs2'


def predict(path_to_images, log_dir):
    id2labels_pickled = os.path.join(log_dir, 'label_names.pkl')
    path_to_weights = os.path.join(log_dir, 'weights.h5')
    inference = InferenceModel(id2label_pickled=id2labels_pickled, path_to_weights=path_to_weights)
    for image_fname in os.listdir(path_to_images):
        full_fname = os.path.join(path_to_images, image_fname)
        try:
            image = prepare_image(Image.open(full_fname).convert('RGB'))
            image = np.asarray(image)
        except Exception as e:
            logging.error(e)
            continue

        gt_label = ''.join(os.path.splitext(image_fname)[0].split('-')[:-1])
        # prediction itself
        pr_label = inference.predict(image)
        logging.info(f"fname={image_fname};\ngt={gt_label}; pr={pr_label};")


class InferenceModel:
    def __init__(self, id2label_pickled, path_to_weights):
        self._id2label = pickle.load(open(id2label_pickled, 'rb'))
        self._model = build_model(N_CLASSES)
        self._model.load_weights(path_to_weights)

    def _predict_proba(self, image):
        img = image
        predicted_ids = self._model.predict(np.expand_dims(preprocess_input(img), 0))[0]
        return predicted_ids

    def predict(self, image):
        predicted_ids = self._predict_proba(image)
        predicted_label = self._id2label[np.argmax(predicted_ids)]
        return predicted_label


def prepare_image(original_image):
    MULTIPLE = 64
    MAX_SIDE = 512
    w, h = original_image.size[:2]
    max_side = max(h, w)
    if max_side > MAX_SIDE:
        if h > w:
            new_h = MAX_SIDE
            new_w = max(1, round(w / h * MAX_SIDE / MULTIPLE)) * MULTIPLE
        else:
            new_w = MAX_SIDE
            new_h = max(1, round(h / w * MAX_SIDE / MULTIPLE)) * MULTIPLE
    else:
        new_h = max(1, round(h / MULTIPLE)) * MULTIPLE
        new_w = max(1, round(w / MULTIPLE)) * MULTIPLE
    # image = cv2.resize(original_image, self._dest_image_size[:2][::-1], interpolation=cv2.INTER_CUBIC)
    image = original_image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image


def build_model(n_classes):
    inp_layer = Input(shape=IMAGE_SIZE)
    cnn = VGG16(include_top=False, weights=None)
    for layer in cnn.layers:
        layer.trainable = True
    x = cnn(inp_layer)
    x_average = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPool2D()(x)
    x = concatenate([x_max, x_average])
    x = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=inp_layer, outputs=x)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    predict(PREDICT_IMAGES_ROOT, PREDICT_LOG_DIR)
