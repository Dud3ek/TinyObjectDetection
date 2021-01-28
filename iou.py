# import used libraries

import tkinter
import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# image functions declarations

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


# model loading

model_name = 'faster_rcnn'
pipeline_config = '/home/dodzio/projects/test/TinyObjectDetection/trained_model/pipeline.config'
model_dir = '/home/dodzio/projects/test/TinyObjectDetection/trained_model'

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-4')).expect_partial()


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


detect_fn = get_model_detection_function(detection_model)

label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

image_dir = '/home/dodzio/projects/test/TinyObjectDetection/datasets/iou/'
image_path = os.path.join(image_dir, 'trial.jpg')
image_np = load_image_into_numpy_array(image_path)

# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(
#     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

# Use keypoints if available in detections
keypoints, keypoint_scores = None, None
if 'detection_keypoints' in detections:
    keypoints = detections['detection_keypoints'][0].numpy()
    keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=500,
    min_score_thresh=.50,
    agnostic_mode=False,
    keypoints=keypoints,
    keypoint_scores=keypoint_scores,
    keypoint_edges=get_keypoint_tuples(configs['eval_config']))

# 4 linijka = detections['detection_scores'][0].numpy()

# print(detections['detection_boxes'][0].numpy())
# print(detections['detection_scores'][0].numpy())
# print(detections['detection_scores'][0].numpy()[1])

# count detections with IoU over 0.5

final_score = detections['detection_scores'][0].numpy()
count = 0
for i in range(len(detections['detection_scores'][0].numpy())):
    if final_score[i] >= 0.5:
        count = count + 1

# add groundtruth boxes

boxes = detections['detection_boxes'][0].numpy()

for pos in range(len(boxes)):
  if pos <= (count-1):
    # print(pos)
    pass
  else:
    for j in range(4):
      boxes[pos,j] = 0

import xml.etree.ElementTree as ET

file_dir = '/home/dodzio/projects/test/TinyObjectDetection/datasets/iou/'
file_path= os.path.join(file_dir, 'coords.xml')
tree = ET.parse(file_path)
root = tree.getroot()
groundboxes = np.zeros((count,4))

k = 0
for xmin in root.findall('./object/bndbox/xmin'):
  groundboxes[k][1] = xmin.text
  k = k + 1

k = 0
for xmax in root.findall('./object/bndbox/xmax'):
  groundboxes[k][3] = xmax.text
  k = k + 1

k = 0
for ymin in root.findall('./object/bndbox/ymin'):
  groundboxes[k][0] = ymin.text
  k = k + 1

k = 0
for ymax in root.findall('./object/bndbox/ymax'):
  groundboxes[k][2] = ymax.text
  k = k + 1

# print(groundboxes)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# ymin - 0
# xmin - 1
# ymax - 2
# xmax - 3

#print(groundboxes)

#print(boxes*1024)

#calculate centers for boxes

boxes_center = np.zeros((count,2))
ground_center = np.zeros((count,2))

for i in range(count):
   boxes_center[i][0] = ((boxes[i][1]*1024) + (boxes[i][3]*1024)) / 2 # x center
   boxes_center[i][1] = ((boxes[i][0]*1024) + (boxes[i][2]*1024)) / 2  # y center

# print(boxes_center)

#calculate groundtruth centers

for i in range(count):
   ground_center[i][0] = ((groundboxes[i][1]) + (groundboxes[i][3])) / 2 # x center
   ground_center[i][1] = ((groundboxes[i][0]) + (groundboxes[i][2])) / 2  # y center

# print(ground_center)

# x1 = boxes_center[0][0]
# x2 = ground_center[3][0]

# y1 = boxes_center[0][1]
# y2 = ground_center[3][1]

# result= ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)

# print(result)

center = np.zeros((1,2))
#print(center)

indexes = np.zeros((count,2))

for i in range(count):

  center[0][0] = boxes_center[i][0]
  center[0][1] = boxes_center[i][1]
  best = 1000
  indexes[i][0] = i

  for j in range(count):
    x1 = center[0][0]
    x2 = ground_center[j][0]
    y1 = center[0][1]
    y2 = ground_center[j][1]

    result = ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
    # print(result)

    if result < best:
      best = result
      indexes[i][1] = j

# print(indexes)

sorted_ground = np.zeros((count,4))

for i in range(count):
  tmp = int(indexes[i][1])
  sorted_ground[i] = groundboxes[tmp]

# print(sorted_ground)
# print(boxes*1024)

# calculate of IoU

iou_results = np.zeros((500))  # has to be as big as number of boxes for consideration

for i in range(count):
    bb1 = {
        'x1': sorted_ground[i][1],
        'y1': sorted_ground[i][0],
        'x2': sorted_ground[i][3],
        'y2': sorted_ground[i][2]
    }

    bb2 = {
        'x1': boxes[i][1] * 1024,
        'y1': boxes[i][0] * 1024,
        'x2': boxes[i][3] * 1024,
        'y2': boxes[i][2] * 1024
    }

    iou = get_iou(bb1, bb2)
    iou_results[i] = iou

# print(iou_results)

sum = 0
for i in range(count):
    sum = sum + iou_results[i]

mean = sum / count

# print(mean)

# print(bb1)
# print(bb2)


image_np = load_image_into_numpy_array(image_path)

# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(
#     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

# Use keypoints if available in detections
keypoints, keypoint_scores = None, None
if 'detection_keypoints' in detections:
  keypoints = detections['detection_keypoints'][0].numpy()
  keypoint_scores = detections['detection_keypoint_scores'][0].numpy()


#count detections with IoU over 0.5
final_score = detections['detection_scores'][0].numpy()
count = 0
for i in range(len(detections['detection_scores'][0].numpy())):
  if final_score[i] >= 0.5:
    count = count + 1


iou_name = detections['detection_classes'][0].numpy() + 1

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      boxes,
      (iou_name + label_id_offset).astype(int),
      iou_results,
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=500,
      min_score_thresh=.30,
      agnostic_mode=False,
      keypoints=keypoints,
      keypoint_scores=keypoint_scores,
      keypoint_edges=get_keypoint_tuples(configs['eval_config']))

# 4 linijka = detections['detection_scores'][0].numpy()

#print(detections['detection_boxes'][0].numpy())
#print(detections['detection_scores'][0].numpy())
#print(detections['detection_scores'][0].numpy()[1])



# add groundtruth boxes

# viz_utils.draw_bounding_box_on_image_array(image_np_with_detections, 424.23047, 505.21033, 473.5871, 563.58154, color='blue', thickness = 5, use_normalized_coordinates = False)

for i in range(len(groundboxes)):
  viz_utils.draw_bounding_box_on_image_array(image_np_with_detections, groundboxes[i][0], groundboxes[i][1], groundboxes[i][2], groundboxes[i][3], color='red', thickness = 3, use_normalized_coordinates = False)


# print('Number of detection boxes to consider ', len(detections['detection_scores'][0].numpy()))
print('Number of detections ', count)
plt.figure(figsize=(12,16))
plt.imshow(image_np_with_detections)
print('Average IoU of boxes on image ', round(mean,2))
plt.savefig("estimate_iou.jpg")