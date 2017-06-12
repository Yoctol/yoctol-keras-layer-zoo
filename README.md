# yoctol-keras-layer-zoo
Customized keras layers used in Yoctol NLU service.

## Features

Our customed layers support mask function in Keras framework. 

The mask function is used to deal with unfixed length of natural language sentences.

* Recurrent Neural Network
  
    * LSTM Peephole 
    * RNN Encoder
    * RNN Decoder
    * Bidirectional RNN Encoder

* Convolutional Neural Network

    * Masked 2D Convolutional layer
    * MaskConv
    * MaskPooling 
    * MaskFlatten 
    * ConvEncoder

## Installation

`pip install yoctol_keras_layer_zoo`

#### Note! We use tensorflow backend while using keras. 
#### Please install tensorflow by yourself. 
#### Either tensorflow-gpu or tensorflow is fine.

  * Install tensorflow with GPU version.

    `pip install tensorflow-gpu`

  * Or install tenserflow with CPU version.

    `pip install tensorflow`

## Test

`python -m unittest`

## Usages

### Recurrent Neural Network

#### LSTM Peephole

Reference: https://en.wikipedia.org/wiki/Long_short-term_memory

The implemented Peephole LSTM outputs its hidden states i.e. h.

 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from yklz import LSTMPeephole
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = LSTMPeephole(
     units=units,
     return_sequences=True
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```

#### LSTM Cell

The Peephole LSTM with another dense layer behind its hidden states.

 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from yklz import LSTMCell
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = LSTMCell(
     units=units,
     return_sequences=True
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```
 
 #### RNN Encoder wrapper
 
 The RNN Encoder encodes sequences into a fixed length vector and 
 pads zero vectors after the encoded vector to provide mask function 
 in the Keras training framework. 
 
 ```
 output tensor shape: (batch_size, timesteps, encoding_size)
 values: [[[encoded_vector], [0,...,0], ..., [0,...,0]], ...]
 ```
 
 You can use any recurrent layer in Keras with our RNN Encoder wrapper.
 
 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers import GRU, Masking
 from yklz import RNNEncoder
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = RNNEncoder(
     GRU(
         units=encoding_size,
     )
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```
 
 To use bidirectional encoding, we provide a customed bidirectional RNN encoder wrapper.

 ```python
 from keras.models import Model, Input
 from keras.layers import LSTM, Masking
 from yklz import BidirectionalRNNEncoder
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = BidirectionalRNNEncoder(
     LSTM(
         units=encoding_size,
     )
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```

#### RNN Decoder

The customed RNN decoder decodes the sequences come from our RNN Encoder.

When our decoder gets zero vector as input, it uses the previous output 
vector as input vector. 

That's why we pad zero vectors after the encoded vector in RNN Encoder.

 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers import LSTM, Masking
 from yklz import RNNEncoder, RNNDecoder
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 encoded_seq = RNNEncoder(
     LSTM(
         units=encoding_size,
     )
 )(masked_inputs)
 outputs = RNNDecoder(
     LSTM(
         units=decoding_size
     )
 )(encoded_seq)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```

### Convolutional Neural Network

#### MaskConv

A masking layer masks 2D, 3D or higher dimensional input tensors.

* Usage

```python
from keras.models import Model, Input
from yklz import MaskConv

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
```

#### MaskPooling

A pooling wrapper supports mask function with Keras pooling layers.

* Usage

```python
from keras.models import Model, Input
from keras.layers.pooling import MaxPool2D

from yklz import MaskConv
from yklz import MaskPooling

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
pooling_outputs = MaskPooling(
    MaxPool2D(
        pooling_kernel,
        pooling_strides,
        pooling_padding,
    ),
    pool_mode='max'
)(masked_inputs)
```

#### 2D ConvNet

Use convolutional neural network to extract text features and make prediction.

* Usage

```python
from keras.models import Model, Input
from keras.layers import Dense
from keras.layers.pooling import MaxPool2D
from yklz import MaskConv, Convolution2D
from yklz import MaskPooling
from yklz import MaskFlatten

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
conv_outputs = Convolution2D(
    filters,
    kernel,
    strides
)(masked_inputs)
pooling_outputs = MaskPooling(
    MaxPool2D(
        pooling_kernel,
        pooling_strides,
        pooling_padding,
    ),
    pool_mode='max'
)(conv_outputs)
flatten_outputs = MaskFlatten()(pooling_outputs)
outputs = Dense(
    label_num
)(flatten_outputs)
model = Model(inputs, outputs)
model.compile('sgd', 'mean_squared_error')
```

### ConvNet to RNN seq2seq model

Encode text features with ConvNet and decode it with RNN.

MaskToSeq is a wrapper transform 2D or 3D mask tensor into timestamp mask tensor.

ConvEncoder transforms a 2D or 3D tensor into a 3D timestamp sequence and mask the sequence 
with the mask tensor from MaskToSeq wrapper.

* Usage
```python
from keras.models import Model, Input
from keras.layers import LSTM
from keras.layers.pooling import MaxPool2D
from yklz import MaskConv, Convolution2D
from yklz import MaskPooling
from yklz import RNNDecoder, MaskToSeq

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
masked_seq = MaskToSeq(
    layer=MaskConv(0.0),
    time_axis=1,
)(inputs)
conv_outputs = Convolution2D(
    filters,
    kernel,
    strides
)(masked_inputs)
pooling_outputs = MaskPooling(
    MaxPool2D(
        pooling_kernel,
        pooling_strides,
        pooling_padding,
    ),
    pool_mode='max'
)(conv_outputs)
encoded = ConvEncoder()(
    [pooling_outputs, masked_seq]
)
outputs = RNNDecoder(
    LSTM(
        units=decoding_size
    )
)(encoded)
model = Model(inputs, outputs)
model.compile('sgd', 'mean_squared_error')
```
