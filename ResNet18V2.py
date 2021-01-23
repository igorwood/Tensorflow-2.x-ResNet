
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dense

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), **kwargs):
        self.strides = strides
        if strides != (1, 1):
            self.shortcut = Conv2D(filters, (1, 1), name='projection', padding='same', use_bias=False)

        self.conv_0 = Conv2D(filters, (3, 3), name='conv_0', strides=strides, padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)
        super(BasicBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.nn.relu(net)

        if self.strides != (1, 1):
            shortcut = tf.nn.avg_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_1(net)

        output = net + shortcut
        return output

class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), projection=False, **kwargs):
        self.strides = strides
        self.projection = projection
        if projection or strides != (1, 1):
            self.shortcut = Conv2D(filters * 4, (1, 1), name='projection', padding='same', use_bias=False)

        self.conv_0 = Conv2D(filters, (1, 1), name='conv_0', padding='same', use_bias=False)
        self.conv_1 = Conv2D(filters, (3, 3), name='conv_1', strides=strides, padding='same', use_bias=False)
        self.conv_2 = Conv2D(filters * 4, (1, 1), name='conv_2', padding='same', use_bias=False)
        self.bn_0 = BatchNormalization(name='bn_0', momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(name='bn_1', momentum=0.9, epsilon=1e-5)
        self.bn_2 = BatchNormalization(name='bn_2', momentum=0.9, epsilon=1e-5)
        super(BottleneckBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        net = self.bn_0(inputs, training=training)
        net = tf.nn.relu(net)

        if self.projection:
                shortcut = self.shortcut(net)
        elif self.strides != (1, 1):
                shortcut = tf.nn.avg_pool2d(net, ksize=(2, 2), strides=(2, 2), padding='SAME')
                shortcut = self.shortcut(shortcut)
        else:
            shortcut = inputs

        net = self.conv_0(net)
        net = self.bn_1(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_1(net)
        net = self.bn_2(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv_2(net)

        output = net + shortcut
        return output

class ResNetV2(tf.keras.models.Model):
    def __init__(self, layer_num, category_num, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        
        block_type = {
          18: 'basic block',
          34: 'basic block',
          50: 'bottlenect block',
          101: 'bottlenect block',
          152: 'bottlenect block'}

        block_num = {
          18: (2, 2, 2, 2),
          34: (3, 4, 6, 3),
          50: (3, 4, 6, 3),
          101: (3, 4, 23, 3),
          152: (3, 4, 36, 3)}

        filter_num = (64, 128, 256, 512)
        
        if block_type[layer_num] == 'basic block':
            self.block = BasicBlock
        else:
            self.block = BottleneckBlock

        self.conv0 = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)

        self.block_collector = []
        for layer_index, (b, f) in enumerate(zip(block_num[layer_num], filter_num), start=1):
            if layer_index == 1:
                if block_type[layer_num] == 'basic block':
                    self.block_collector.append(self.block(f, name='conv1_0'))
                else:
                    self.block_collector.append(self.block(f, projection=True, name='conv1_0'))
            else:
                self.block_collector.append(self.block(f, strides=(2, 2), name='conv{}_0'.format(layer_index)))

            for block_index in range(1, b):
                self.block_collector.append(self.block(f, name='conv{}_{}'.format(layer_index, block_index)))

        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)
        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(category_num, name='fully_connected', activation='softmax', use_bias=False)

    def call(self, inputs, training):
        net = self.conv0(inputs)
        net = tf.nn.max_pool2d(net, ksize=(3, 3), strides=(2, 2), padding='SAME')

        for block in self.block_collector:
            net = block(net, training)
        net = self.bn(net, training)
        net = tf.nn.relu(net)

        net = self.global_average_pooling(net)
        net = self.fc(net)
        #print('fully connected', net.shape)
        return net
        
def ResNet18V2(category_num=1000):
  return ResNetV2(layer_num=18, category_num=1000)
  
