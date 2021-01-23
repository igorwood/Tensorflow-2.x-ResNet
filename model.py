
import os, pathlib, PIL
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model

from ResNet18 import ResNet18
from ResNet18V2 import ResNet18V2

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2


class ResNet(Model):
  def __init__(self, data_shape=(224, 224, 3), resnet_version=1, resnet_layer_number=50, num_classes=1000):
    super(ResNet, self).__init__()
    
    weights = None
    if num_classes == 1000 and data_shape == (224, 224, 3):
      weights = 'imagenet'
      
    self.resnet_version = resnet_version
    
    self.data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip(
          "horizontal", 
          input_shape=data_shape),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )
    
    self.rescaling = layers.experimental.preprocessing.Rescaling(1./255)
    
    def preprocess_input(x, data_format=None):
      from tensorflow.keras.applications import imagenet_utils
      return imagenet_utils.preprocess_input(
      x, data_format=data_format, mode='tf')
      #return x
      
    self.preprocess_input = preprocess_input
    
    if resnet_layer_number == 18:
      if resnet_version == 1:
        self.resnet = ResNet18(category_num=num_classes)
      else:
        self.resnet = ResNet18V2(category_num=num_classes)
    elif resnet_layer_number == 50:
      if resnet_version == 1:
        self.resnet = ResNet50(weights=weights, input_shape=data_shape, classes=num_classes)
      else:
        self.resnet = ResNet50V2(weights=weights, input_shape=data_shape, classes=num_classes)
    elif resnet_layer_number == 101:
      if resnet_version == 1:
        self.resnet = ResNet101(weights=weights, input_shape=data_shape, classes=num_classes)
      else:
        self.resnet = ResNet101V2(weights=weights, input_shape=data_shape, classes=num_classes)
    elif resnet_layer_number == 152:
      if resnet_version == 1:
        self.resnet = ResNet152(weights=weights, input_shape=data_shape, classes=num_classes)
      else:
        self.resnet = ResNet152V2(weights=weights, input_shape=data_shape, classes=num_classes)
      
    self.build((None,) + data_shape)

  def call(self, x):
    x = self.data_augmentation(x)
    x = self.rescaling(x)
    x = self.preprocess_input(x)
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    x = self.resnet(x)
    return x


class ResNetWork():
  def __init__(self, args):
    # dataset
    train_data_dir = pathlib.Path(args.train_dataset_path)
    test_data_dir = pathlib.Path(args.test_dataset_path)
    
    self.image_height = args.image_height
    self.image_width = args.image_width
    data_shape = (args.image_height, args.image_width, 3)
    batch_size = args.batchsize
    
    pretrain_model_path_or_dir = args.pre_train_model_path_dir
    
    # create model
    self.model = ResNet(
        data_shape = data_shape,
        resnet_version=args.resnet_version,
        resnet_layer_number=args.resnet_layer_number,
        num_classes=args.classes)
        
    if os.path.exists(pretrain_model_path_or_dir):
      if args.use_whole_network_model:
        dir = pretrain_model_path_or_dir
        self.model = keras.models.load_model(dir)
        print("Whole network load from {} dir".format(dir))
      else:
        path = pretrain_model_path_or_dir
        self.model.load_weights(path)
        print("Network model load from {}".format(path))
    
    # Optimization
    self.learning_rate = args.lr
    self.epochs = args.epochs
    
    if args.opt_type == 'Adam':
      self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.lr)
    elif args.opt_type == 'SGD':
      self.optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=args.momentum)
    elif args.opt_type == 'Adadelta':
      self.optimizer = tf.keras.optimizers.Adadelta(
        learning_rate=args.lr)
    elif args.opt_type == 'Adamax':
      self.optimizer = tf.keras.optimizers.Adamax(
        learning_rate=args.lr)
    elif args.opt_type == 'Ftrl':
      self.optimizer = tf.keras.optimizers.Ftrl(
        learning_rate=args.lr)
    elif args.opt_type == 'Nadam':
      self.optimizer = tf.keras.optimizers.Nadam(
        learning_rate=args.lr)
    else:
      self.optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=args.lr)
        
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # get the data set
    train_image_count = 0
    train_image_count += len(list(train_data_dir.glob('*/*.jpg')))
    train_image_count += len(list(train_data_dir.glob('*/*.JPEG')))
    print("train image number:", train_image_count)
    
    test_image_count = 0
    test_image_count += len(list(test_data_dir.glob('*/*.jpg')))
    test_image_count += len(list(test_data_dir.glob('*/*.JPEG')))
    print("Test image number:", test_image_count)
    
    # train dataset
    self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      train_data_dir,
      #subset="training",
      seed=123,
      image_size=(args.image_height, args.image_width),
      batch_size=batch_size)
    self.class_names = self.train_ds.class_names
    self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
      
    # valid/test dataset
    self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      test_data_dir,
      #subset="validation",
      seed=123,
      image_size=(args.image_height, args.image_width),
      batch_size=batch_size)
    self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    self.test_loss = tf.keras.metrics.Mean(name='valid_loss')
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='vaild_accuracy')
  
  @tf.function
  def train_step(self, images, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(images)
      loss = self.loss_object(labels, predictions)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    self.train_loss(loss)
    self.train_accuracy(labels, predictions)
  # [end train_step]
    
  @tf.function
  def test_step(self, images, labels):
    predictions = self.model(images)
    t_loss = self.loss_object(labels, predictions)

    self.test_loss(t_loss)
    self.test_accuracy(labels, predictions)
  # [end test_step]
    
  def train(self):
    # Model summary
    self.model.summary()
    
    for epoch in range(self.epochs):
    
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      self.test_loss.reset_states()
      self.test_accuracy.reset_states()
      
      try:
        with tqdm(self.train_ds, ncols=80) as t:
          for images, labels in t:
            self.train_step(images, labels)
            template = '[Train\t Epoch {}] Loss: {:.4f}, Accuracy: {:.4f}'
            template = template.format(epoch+1, self.train_loss.result(), self.train_accuracy.result()*100)
            t.set_description(desc=template)
      except KeyboardInterrupt:
        t.close()
        raise

      try:
        with tqdm(self.test_ds, ncols=80) as t:
          for test_images, test_labels in t:
            self.test_step(test_images, test_labels)
            template = '[Test\t Epoch {}] Loss: {:.4f}, Accuracy: {:.4f}'
            template = template.format(epoch+1, self.test_loss.result(), self.test_accuracy.result()*100)
            t.set_description(desc=template)
      except KeyboardInterrupt:
        t.close()
        raise
  # [end train]
        
  def saveModel(self, path_or_dir, mode='save_weight'):
    if mode == 'save_weight':
      path = path_or_dir
      self.model.save_weights(path)
      print("Network model save to {}".format(path))
    elif mode == 'whole_network':
      dir = path_or_dir
      self.model.save(dir)
      print("Whole network save to {} dir".format(dir))
  # [end saveModel]
  
  def test(self, args):
    if not os.path.exists(args.test_image):
      return
      
    image_path = args.test_image
      
    img = keras.preprocessing.image.load_img(
      image_path, target_size=(
        args.image_height,
        args.image_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
        
    import numpy as np
    print("{} most likely belongs to {} with a {:.2f} percent confidence.".format(image_path, self.class_names[np.argmax(score)], 100 * np.max(score)))
  # [end test]
    
    
    
    
    