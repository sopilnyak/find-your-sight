import os
import sys
import numpy as np
import cv2
import keras
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from vgg_places365 import VGG16_Places365
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import itertools
import logging
import pickle
import threading

IMAGES_ROOT = '/home/andreyzharkov/data/data_full'
N_CLASSES = 29

IMAGE_SIZE = (None, None, 3)
BATCH_SIZE = 1
N_WORKERS = 4
LOG_DIR = 'logs2'
SAVE_DIR = os.path.join(LOG_DIR, 'embeddings')

# change them
PREDICT_IMAGES_ROOT = IMAGES_ROOT
PREDICT_LOG_DIR = LOG_DIR

np.random.seed(0)


class InferenceModel:
    def __init__(self, id2label_pickled, path_to_weights=os.path.join(LOG_DIR, 'model.h5')):
        self._id2label = pickle.load(open(id2label_pickled, 'rb'))
        self._image_preprocessor = ImagePreprocessor(IMAGE_SIZE, is_training=False)
        self._model = build_model(N_CLASSES)
        self._model.load_weights(path_to_weights)

    def _predict_proba(self, image):
        input_img = self._image_preprocessor.prepare_image(image)
        predicted_ids = self._model.predict(np.expand_dims(input_img, 0))[0]
        return predicted_ids

    def predict(self, image):
        predicted_ids = self._predict_proba(image)
        predicted_label = self._id2label[np.argmax(predicted_ids)]
        return predicted_label


def save_model_weights(model, path_to_weights_dir):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        lw_dir = os.path.join(path_to_weights_dir, f"layer{i:03d}")
        os.makedirs(lw_dir, exist_ok=True)
        for j, w in enumerate(weights):
            np.save(os.path.join(lw_dir, f"{j:03d}"), w)


def train(path_to_images):
    model = build_model(n_classes=N_CLASSES)
    save_model_weights(model, os.path.join(LOG_DIR, 'weights_numpy'))

    model.summary()
    optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    train_generator = BatchGenerator(reader=Reader(path_to_images, 'train'))
    valid_generator = BatchGenerator(reader=Reader(path_to_images, 'valid'))
    assert np.all(train_generator._reader._label_names == valid_generator._reader._label_names), \
        f"train={train_generator._reader._label_names},\n\nvalid={valid_generator._reader._label_names}"

    saving_callback = keras.callbacks.ModelCheckpoint(os.path.join(LOG_DIR, "model_{epoch:03d}.h5"),
                                                      save_weights_only=True)
    saving_callback2 = keras.callbacks.ModelCheckpoint(os.path.join(LOG_DIR, "model.h5"),
                                                       save_weights_only=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=False, write_images=True)
    callbacks = [saving_callback, saving_callback2, tensorboard]

    logging.info(f"Training on {train_generator.get_images_count()} images, "
                 f"validating on {valid_generator.get_images_count()}")

    model.fit_generator(generator=train_generator.generate(BATCH_SIZE, is_training=False),
                        steps_per_epoch=train_generator.get_steps_per_epoch(BATCH_SIZE),
                        epochs=5, verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator.generate(BATCH_SIZE, is_training=False),
                        validation_steps=valid_generator.get_steps_per_epoch(BATCH_SIZE),
                        max_queue_size=30,
                        workers=N_WORKERS,
                        use_multiprocessing=(N_WORKERS > 1),
                        shuffle=True,
                        )


def predict(path_to_images, log_dir):
    id2labels_pickled = os.path.join(log_dir, 'label_names.pkl')
    # path_to_weights = os.path.join(log_dir, 'model.h5')
    path_to_weights = os.path.join(log_dir, 'weights.h5')
    inference = InferenceModel(id2label_pickled=id2labels_pickled,path_to_weights=path_to_weights)
    for image_fname in os.listdir(path_to_images):
        full_fname = os.path.join(path_to_images, image_fname)
        image = cv2.imread(full_fname)
        gt_label = ''.join(os.path.splitext(image_fname)[0].split('-')[:-1])
        # prediction itself
        pr_label = inference.predict(image)
        logging.info(f"fname={image_fname};\ngt={gt_label}; pr={pr_label};\n")


def get_data(path_to_images, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model = build_model(n_classes=N_CLASSES, include_top=False)
    model.summary()

    train_generator = BatchGenerator(reader=Reader(path_to_images, 'train'))
    valid_generator = BatchGenerator(reader=Reader(path_to_images, 'valid'))

    logging.info("getting features for train...")
    get_features(model, train_generator, save_fn_pref='train')
    logging.info("getting features for valid...")
    get_features(model, valid_generator, save_fn_pref='valid')


def get_features(model, generator, save_fn_pref=''):
    all_embeddings = []
    all_labels = []
    for images, labels in generator.generate(batch_size=BATCH_SIZE, is_training=False):
        embeddings = model.predict(images)
        all_labels.extend(labels)
        all_embeddings.extend(embeddings.tolist())
        if len(all_labels) >= generator.get_images_count():
            break
    logging.info(f"{len(all_labels)}/{generator.get_images_count()} images processed")
    emb_fname = os.path.join(LOG_DIR, f'{save_fn_pref}_embeddings')
    labels_fname = os.path.join(LOG_DIR, f'{save_fn_pref}_labels')
    np.save(emb_fname, np.array(all_embeddings))
    np.save(labels_fname, np.array(all_labels))
    # pickle.dump(np.array(all_embeddings), open(emb_fname, 'wb'))
    # pickle.dump(np.array(all_labels), open(labels_fname, 'wb'))


class Reader:
    def __init__(self, images_root, split='train'):
        assert split in ['train', 'valid']
        self._images_root = images_root
        self._read_markup(split)

    def get_image(self, image_id):
        img = cv2.imread(os.path.join(self._images_root, self._image_names[image_id]))
        assert img is not None, f"{self._image_names[image_id]} bad image"
        return img

    def get_image_name(self, image_id):
        return self._image_names[image_id]

    def get_label(self, image_id):
        return self._image_labels[image_id]

    def get_image_ids(self):
        return np.arange(0, len(self._image_names), dtype=np.int)

    def get_label_id(self, label_name):
        return self._label_to_id[label_name]

    def _read_markup(self, split):
        self._image_names = os.listdir(self._images_root)
        self._get_split(split)
        self._image_labels = [''.join(os.path.splitext(n)[0].split('-')[:-1]) for n in self._image_names]
        self._label_names = np.unique(self._image_labels)
        self._label_to_id = dict((l, i) for i, l in enumerate(self._label_names))
        pickle.dump(self._label_names, open(os.path.join(LOG_DIR, 'label_names.pkl'), 'wb'))

    def _get_split(self, split):
        selected_image_names = []
        logging.info("Generating split for {split}")
        for label, group in itertools.groupby(sorted(self._image_names), key=lambda n: ''.join(os.path.splitext(n)[0].split('-')[:-1])):
            group = list(group)
            val_p = 0.1
            min_valid_images = 5
            n_valid_images = max(int(round(len(group)) * val_p), min_valid_images)
            # TODO random
            if split == 'train':
                label_data_split = group[:-n_valid_images]
            elif split == 'valid':
                label_data_split = group[-n_valid_images:]
            else:
                raise ValueError()
            selected_image_names.extend(label_data_split)
            logging.info(f"for {label} in phase {split} selected {len(label_data_split)}")
        self._image_names = selected_image_names


class ImagePreprocessor:
    def __init__(self, dest_size, is_training=True):
        self._dest_image_size = dest_size
        self._is_training = is_training
        self._aug_seq = self._get_seq()

    def prepare_images(self, original_images):
        images = [self.prepare_image(img) for img in original_images]
        if self._is_training:
            images = self._aug_seq.augment_images(images)
        return images

    @staticmethod
    def prepare_image(original_image):
        # logging.info(f"original image of type {type(original_image)}")
        MULTIPLE = 64
        MAX_SIDE = 512
        h, w = original_image.shape[:2]
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
        image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image

    def _get_seq(self):
        ia.seed(1)

        p_original_image = 0.5
        seq = iaa.Sometimes(p_original_image, iaa.SomeOf([1, None], [
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8),
                order=[0, 1],
                mode='constant'
            ),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
            iaa.CoarseSaltAndPepper(p=0.05, per_channel=True, size_px=(7, 15))
        ], random_order=True))  # apply augmenters in random order

        # TODO fixed crop

        return seq


class BatchGenerator:
    def __init__(self, reader):
        self._reader = reader
        self._image_ids = reader.get_image_ids()
        self._lock = threading.Lock()
        self._curr_idx = 0
        logging.info(f"image ids: {self._image_ids}")

    def generate(self, batch_size, is_training=True, save_images=False):
        img_processor = ImagePreprocessor(IMAGE_SIZE, is_training=is_training)
        max_ind = len(self._image_ids) // batch_size + 1
        while True:
            if is_training:
                with self._lock:
                    np.random.shuffle(self._image_ids)

            while True:
                with self._lock:
                    if self._curr_idx >= max_ind:
                        self._curr_idx = 0
                    start_idx = self._curr_idx
                    self._curr_idx += 1
                images = [
                    self._reader.get_image(image_id) for image_id in self._image_ids[start_idx:start_idx + batch_size]]
                labels = [
                    self._reader.get_label(image_id) for image_id in self._image_ids[start_idx:start_idx + batch_size]]
                images = np.array([img_processor.prepare_image(img) for img in images])  # TODO too slow
                images = img_processor.prepare_images(images)

                if save_images:
                    tmp = os.path.join(LOG_DIR, 'tmp')
                    os.makedirs(tmp, exist_ok=True)
                    for img_name, img in zip(
                            [
                                self._reader.get_image_name(image_id) for image_id in self._image_ids[start_idx:start_idx + batch_size]
                            ], images):
                        cv2.imwrite(os.path.join(tmp, img_name), img)

                images = np.array([preprocess_input(img) for img in images])  # TODO this too
                labels = np.array([self._reader.get_label_id(l) for l in labels])
                yield images, labels

    def get_images_count(self):
        return self._image_ids.shape[0]

    def get_steps_per_epoch(self, batch_size):
        return max(1, self.get_images_count() // batch_size)


def build_model(n_classes, include_top=True):
    inp_layer = Input(shape=IMAGE_SIZE)
    # cnn = VGG16(include_top=False, weights='imagenet')
    cnn = VGG16_Places365(include_top=False, weights='places')
    for layer in cnn.layers:
        layer.trainable = True
    x = cnn(inp_layer)
    x_average = keras.layers.GlobalAveragePooling2D()(x)
    x_max = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.concatenate([x_max, x_average])
    # x = Dense(100, activation='relu')(x)
    # x = Dense(100, activation='relu')(x)
    if include_top:
        x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp_layer, outputs=x)
    if include_top:
        model.layers[-1].set_weights([
            np.load(os.path.join(LOG_DIR, 'W.npy')).T,
            np.load(os.path.join(LOG_DIR, 'b.npy'))
        ])
        model.save_weights(os.path.join(LOG_DIR, 'weights.h5'))
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train(IMAGES_ROOT)
    # predict(PREDICT_IMAGES_ROOT, PREDICT_LOG_DIR)
    # get_data(IMAGES_ROOT)
