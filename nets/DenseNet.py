from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import concatenate
from keras.layers import Add
from keras import Model

def _conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def _dense_block(x, blocks, growth_rate, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = _conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x

def prepareInput(inputs):
    input_layer, x = inputs
    orig = input_layer[:, :, :, 0:3]
    return concatenate([orig, x])

def RefineDenseNet(model=None,
                   blocks=4,
                   growth_rate=3,
                   name='dense',
                   train_prev=False
                   ):
    """Add to existing model a dense net layer to refine edge cases.
    Input: 
        model: an encoder-decoder architecture,
        blocks: number of conv block in densenet layer
        growth_rate: growth rate of kernel in conv layers
        name: name
        train_prev: boolean whether to train the previous encoder-decoder model or not
    Return: encoder-decoder model + densenet refine layer"""

    inputs = model.input
    img_height = inputs.shape[1].value
    img_width = inputs.shape[2].value
    outputs = model.output
    
    if (train_prev == False):
        for layer in model.layers:
            layer.trainable = False
    elif (train_prev == True):
        for layer in model.layers:
            layer.trainable = True
    # Concat mask output with input
    x = Lambda(prepareInput, name='concat_input')([inputs, outputs])
    # add dense block
    x = _dense_block(x, blocks, growth_rate, name)
    x = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='linear')(x)
    x = Add()([x, outputs])
    x = Activation('sigmoid', name='refine')(x)

    # Create model
    model = Model(inputs=[inputs], outputs=[x])

    return model


