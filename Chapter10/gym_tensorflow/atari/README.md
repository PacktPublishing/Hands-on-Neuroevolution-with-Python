Notice
-----------------
The ALE/atari-py is not part of deep-neuroevolution.
This folder provides the instructions and sample code if you are interested in running the ALE.
It depends on atari-py. atari-py is licensed under GPLv2.

Instructions
-----------------

The first thing to do is clone the atari-py repository into the `gym_tensorflow` folder using
```
git clone https://github.com/yaricom/atari-py.git
```

Now you can build the library with `cd ./atari-py && make`.

## Building GYM Tensorflow

Building `cd ../..gym_tensorflow && make` should give you access to the Atari games as a set of TensorFlow ops.