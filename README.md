# yoctol-keras-layer-zoo
Customized keras layers used in Yoctol NLU service.

## Features

Our customized layers support mask function in Keras framework. 

The mask function is used to deal with unfixed length of natural language sentences.

* Recurrent Neural Network
  
    * LSTM Peephole 
    * RNN Cell
    * RNN Encoder
    * RNN Decoder
    * Bidirectional RNN Encoder

* Convolutional Neural Network

    * Mask ConvNet
    * MaskConv
    * Mask Pooling 
    * Mask Flatten 
    * ConvEncoder

## Installation

`pip install yoctol_keras_layer_zoo`

#### Note! We use tensorflow backend while using keras. 

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

#### RNN Cell

The RNNCell add another Dense layer behind its recurrent layer.

 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from keras.layers import LSTM, Dense
 from yklz import RNNCell
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = RNNCell(
     LSTM(
         units=hidden_units,
         return_sequences=True
     ),
     Dense(
         units=units
     ),
     dense_dropout=0.1
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

Note that your encoding size should be the same with the decoding size.

The decoder could also decode sequences with different length by specifying
your decoding length.

 * Usage:
 
  * Auto-encoder: decoding sequences whose length and mask are same with
  the input sequences.

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

  * Seq2Seq: decoding sequences whose length is different with input sequences

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
     ),
     time_steps=decoding_length
 )(encoded_seq)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```

### Convolutional Neural Network

Note we use channel last data format with our ConvNet layers.

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
#### MaskConvNet

A wrapper supports mask function with Keras convolutional layers.

* Usage

```python
from keras.models import Model, Input
from keras.layers import Conv2D

from yklz import MaskConv
from yklz import MaskConvNet

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
conv_outputs = MaskConvNet(
    Conv2D(
        filters,
        kernel,
        strides=strides
    )
)(masked_inputs)
```

#### 2D masked ConvNet model

Use convolutional neural network to extract text features and make prediction.

* Usage

```python
from keras.models import Model, Input
from keras.layers import Dense
from keras.layers.pooling import MaxPool2D
from keras.layers import Conv2D

from yklz import MaskConv
from yklz import MaskConvNet
from yklz import MaskPooling
from yklz import MaskFlatten

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)

conv_outputs = MaskConvNet(
    Conv2D(
        filters,
        kernel,
        strides=strides
    )
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

#### Auto-encoder example

* Usage
```python
from keras.models import Model, Input
from keras.layers import LSTM
from keras.layers.pooling import MaxPool2D
from keras.layers import Conv2D, Dense

from yklz import MaskConv
from yklz import MaskConvNet
from yklz import MaskPooling
from yklz import RNNDecoder, MaskToSeq
from yklz import RNNCell

inputs = Input(shape=(seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
masked_seq = MaskToSeq(
    layer=MaskConv(0.0),
    time_axis=1,
)(inputs)

conv_outputs = MaskConvNet(
    Conv2D(
        filters,
        kernel,
        strides=strides
    )
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
    RNNCell(
        LSTM(
            units=hidden_size
        ),
        Dense(
            units=decoding_size
        )
    )
)(encoded)
model = Model(inputs, outputs)
model.compile('sgd', 'mean_squared_error')
```

#### Seq2Seq example

* Usage
```python
from keras.models import Model, Input
from keras.layers import LSTM
from keras.layers.pooling import MaxPool2D
from keras.layers import Conv2D
from keras.layers import Masking, Dense

from yklz import MaskConv
from yklz import MaskConvNet
from yklz import MaskPooling
from yklz import RNNDecoder
from yklz import RNNCell

inputs = Input(shape=(input_seq_max_length, word_embedding_size, channel_size))
masked_inputs = MaskConv(0.0)(inputs)
masked_seq = MaskToSeq(
    layer=MaskConv(0.0),
    time_axis=1,
)(inputs)

conv_outputs = MaskConvNet(
    Conv2D(
        filters,
        kernel,
        strides=strides
    )
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
    RNNCell(
        LSTM(
            units=hidden_size
        ),
        Dense(
            units=decoding_size
        )
    ),
    time_steps=decoding_length
)(encoded)
model = Model(inputs, outputs)
model.compile('sgd', 'mean_squared_error')
```

For more examples you could visit our seq2vec repository.

https://github.com/Yoctol/seq2vec

The seq2vec repository contains auto-encoder models which encode 
sequences into fixed length feature vector.
