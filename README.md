# retraining-tensorflow-model

This repo special for retraining TensorFlow Model based on Inception v3.

Details about google Inception v3 can be found from [here](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf).

## Steps

- Ubuntu 16.04
- install [tensorflow](https://github.com/tensorflow/tensorflow/#installation)
- install git

```
$ sudo apt-get update
$ sudo apt-get install git
```

### Prepare data

Moving your folders to **data** folder. Each of your folders should contain some image files to be trained or validation or test and named by number.

The structure of **data** like the following:


```
- data
    - iphone
        - 0.jpg
        - 1.jpg
        - 2.jpg
        - ...
    - samsung
        - 0.jpg
        - 1.jpg
        - 2.jpg
        - ...
    - mi
        - 0.jpg
        - 1.jpg
        - 2.jpg
        - ...
```


### Download retraining script

```
$ curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.2/tensorflow/examples/image_retraining/retrain.py
```

### Start tensorboard in background

```
$ tensorboard --logdir training_summaries &
```

>This command will fail with the following error if you already have a tensorboard running: 
`ERROR:tensorflow:TensorBoard attempted to bind to port 6006, but it was already in use` 
You can kill all existing TensorBoard instances with: `$ pkill -f "tensorboard"`

### Start the retraining script

```
$ python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=data
```

>If you not set `--how_many_training_steps=500`，the default value will be 4000.

```
$ python retrain.py \
  --bottleneck_dir=bottlenecks \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=data
```

After the bottleneck for each image is computed and stored in bottlenecks. You can use tensorboard to wath what's going on while training by type `localhost:6006` into the address of your brower like google Chrome brower.

### Conclude

The retraining script will write out a version of the Inception v3 network with a final layer retrained to your categories to `retrained_graph.pb` and a text file containing the labels to `retrained_labels.txt`.

### Last but not least

Testing your retaining tensorflow model by typing some commands like the following:

```
$ curl -L https://goo.gl/3lTKZs > label_image.py
$ python label_image.py data/iphone/0.jpg
```

What in the **label_image.py**:

```
import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

```


## Optional Part

### Optimize for inference

```
$ python -m tensorflow.python.tools.optimize_for_inference \
  --input=retrained_graph.pb \
  --output=optimized_graph.pb \
  --input_names="Cast" \
  --output_names="final_result"
```

### Quantize the network weights

```
$ python -m scripts.quantize_graph \
  --input=optimized_graph.pb \
  --output=rounded_graph.pb \
  --output_node_names=final_result \
  --mode=weights_rounded
```


## Reference

1. [TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0)
2. [TensorFlow For Poets 2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#0)
3. [Android TensorFlow support](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)
4. [TensorFlow Android Camera Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)
