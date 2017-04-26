# yoctol-keras-layer-zoo
Some customized keras layers used in Yoctol NLU.

## Install

`pip install yoctol_keras_layer_zoo`

### Note! We use tensorflow backend while using keras. 
### Please install tensorflow by yourself. 
### Either tensorflow-gpu or tensorflow is fine.

  * Install tensorflow with GPU version.

    `pip install tensorflow-gpu`

  * Or install tenserflow with CPU version.

    `pip install tensorflow`

## Test

`python -m unittest`

## Customed Layers

### LSTM Peephole

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

### LSTM Cell

The Peephole LSTM with another dense layer behind its hidden states.

 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from yklz import LSTMCell
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = LSTMCell(
     output_units=output_units,
     return_sequences=True
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```
 
 ### LSTM Encoder
 
 The LSTM Encoder encodes sequences into a fixed length vector and 
 pads zero vectors after the encoded vector to provide mask function 
 in the Keras training framework. 
 
 ```
 output tensor shape: (batch_size, timesteps, encoding_size)
 values: [[[encoded_vector], [0,...,0], ..., [0,...,0]], ...]
 ```
 
 We use our customed LSTMCell in LSTMEncoder.
 
 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from yklz import LSTMEncoder
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = LSTMEncoder(
     output_units=encoding_size,
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```
 
 To use bidirectional LSTM, we provide a customed bidirectional encoder wrapper.

 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from yklz import LSTMEncoder, Bidirectional_Encoder
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 outputs = Bidirectional_Encoder(
     LSTMEncoder(
         output_units=encoding_size,
     )
 )(masked_inputs)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```

### LSTM Decoder

The customed LSTM decoder decodes the sequences come from our LSTMEncoder.

When our decoder gets zero vector as input, it uses the previous output 
vector as input vector. 

That's why we pad zero vectors after the encoded vector in LSTMEncoder.

 * Usage:
 
 ```python
 from keras.models import Model, Input
 from keras.layers.core import Masking
 from yklz import LSTMEncoder, LSTMDecoder
 
 inputs = Input(shape=(max_length, feature_size))
 masked_inputs = Masking(0.0)(inputs)
 encoded_seq = LSTMEncoder(
     output_units=encoding_size,
 )(masked_inputs)
 outputs = LSTMDecoder(
     output_units=decoding_size
 )(encoded_seq)
 model = Model(inputs, outputs)
 model.compile('sgd', 'mean_squared_error')
 ```
