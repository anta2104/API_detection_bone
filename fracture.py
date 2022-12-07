import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf
# model = tf.saved_model.load("./my_model_centernet_1/saved_model")

# detect_fn = model.signatures['serving_default']

# import json 
# def load_labels_map(label_url):
#     # read JSON file
#     a =  open(label_url)
#     data = json.load(a)
#     if data:
#         return data['labels']
#     return []
# label_maps = load_labels_map('label_map_bone.pbtxt')

def process_output(data, threshold, targetSize, labels_map):
    """
    process output of model
    Parameters:
        type (str): type of model
        data (numpy array): output of model
        threshold (float): threshold of model
        targetSize (dict): target size of image
    Returns:
        result (dict): list bounding boxes
    """

    scores, boxes, classes = None, None, None
    label_map = None
    scores = list(data['detection_scores'][0])
    boxes = list(data['detection_boxes'][0])
    classes = list(data['detection_classes'][0])
    label_map = labels_map
    results = {}
    for i in range(len(scores)):
        if scores[i] > threshold:
            label = label_map[int(classes[i]) - 1]
            if label in results:
                results[label].append([
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h'],scores[i]])
            else:
                results[label] = [[
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h'],scores[i]]]
    
    return results


# img = cv2.imread('D:/ObjectDetection_TensorFlow2/2.jpg')
# img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
# input_tensor = tf.convert_to_tensor(img)
# input_tensor = input_tensor[tf.newaxis, ...]

# results = detect_fn(input_tensor)

# targetSize = { 'w': 0, 'h': 0 }
# targetSize['h'] = img.shape[0]
# targetSize['w'] = img.shape[1]

# output = process_output(results, 0.5, targetSize, label_maps)

# img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# boxes = np.array(output['fracture'])
# font = cv2.FONT_HERSHEY_PLAIN
# height, width, channels = img.shape
# for i in range(len(boxes)):
#             x, y, w, h = boxes[i]
#             # x = x /width
#             # y = y / height
#             # w = w / width
#             # h = h / height
#             label = 'Fracture'
#             color = (0,0,255)
#             #score = str(round(confidences[i],4))
#             cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), color, 2)
#             cv2.putText(img, label, (int(x), int(y - 10)), font, 1, color, 1)
#             #cv2.putText(img, score, (x, y + 50), font, 2, color, 2)

# cv2.imwrite('test.jpg' , img)
# key = cv2.waitKey(0)