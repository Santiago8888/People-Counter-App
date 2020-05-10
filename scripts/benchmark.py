import tensorflow as tf
import numpy as np
import time
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Initializing!')

def openGraph(path):
  graph = tf.Graph()
  graphDef = tf.compat.v1.GraphDef()

  with open(path, "rb") as graphFile:
    graphDef.ParseFromString(graphFile.read())

  with graph.as_default():
    tf.import_graph_def(graphDef)

  return graph

graph = openGraph('frozen_inference_graph.pb')
input = graph.get_tensor_by_name('import/image_tensor:0')

outputs= [
  "import/num_detections:0",
  "import/detection_classes:0",
  "import/detection_scores:0",
  "import/detection_boxes:0",
  "import/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/TensorArrayWrite/TensorArrayWriteV3:0",
  "import/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/NextIteration_1:0",

  "import/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/Merge:0",
  "import/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/Merge:1",
  "import/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Merge:0",
  "import/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Merge:1"
]

image = cv2.imread('image.jpeg')
image = cv2.resize(image, (300, 300))
image = np.copy(image)
image = image.astype('uint8')


image = cv2.resize(image, (300, 300))
image = image.reshape(1, 300, 300, 3)

with tf.compat.v1.Session(graph=graph) as sess:
#  layers = sess.graph.get_operations()
#  layers = [m.values() for m in layers]
#  for layer in layers:
#    if len(layer) > 1: print(layer[1], len(layer))

  for output in outputs:
    start = time.time()
    sess.run(output, feed_dict={input: image})
    end = time.time()
    print(output, end - start)

#    print(output, sess.run(output, feed_dict={input: image}))














