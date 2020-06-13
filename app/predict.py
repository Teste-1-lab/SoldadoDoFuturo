# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
"""Sample prediction script for TensorFlow 2.x."""
import sys
import tensorflow as tf
import numpy as np
import cv2
import math
import base64
import zmq

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://192.168.1.85:5555')

MODEL_FILENAME = '../model.pb'
LABELS_FILENAME = '../labels.txt'
IMAGE_FILENAME = "../Biblioteca de Imagens/21.jpg"

class ObjectDetection(object):
    """Class for Custom Vision's exported object detection model
    """

    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
    IOU_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = 512 * 512
    
    def __init__(self, labels, prob_threshold=0.5, max_detections = 20):
        """Initialize the class

        Args:
            labels ([str]): list of labels for the exported model.
            prob_threshold (float): threshold for class probability.
            max_detections (int): the max number of output results.
        """

        assert len(labels) >= 1, "At least 1 label is required"

        self.labels = labels
        self.prob_threshold = prob_threshold
        self.max_detections = max_detections
    
    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    
    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        """Remove overlapping bouding boxes
        """
        assert len(boxes) == len(class_probs)

        max_detections = min(max_detections, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)

        areas = boxes[:, 2] * boxes[:, 3]

        selected_boxes = []
        selected_classes = []
        selected_probs = []

        while len(selected_boxes) < max_detections:
            # Select the prediction with the highest probability.
            i = np.argmax(max_probs)
            if max_probs[i] < self.prob_threshold:
                break

            # Save the selected prediction
            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])

            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]

            # Get overlap between the 'box' and 'other_boxes'
            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[0] + box[2], other_boxes[:, 0] + other_boxes[:, 2])
            y2 = np.minimum(box[1] + box[3], other_boxes[:, 1] + other_boxes[:, 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            # Calculate Intersection Over Union (IOU)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)

            # Find the overlapping predictions
            overlapping_indices = other_indices[np.where(iou > self.IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)

            # Set the probability of overlapping predictions to zero, and udpate max_probs and max_classes.
            class_probs[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(class_probs[overlapping_indices], axis=1)
            max_classes[overlapping_indices] = np.argmax(class_probs[overlapping_indices], axis=1)

        assert len(selected_boxes) == len(selected_classes) and len(selected_boxes) == len(selected_probs)
        return selected_boxes, selected_classes, selected_probs
    
    
    def _extract_bb(self, prediction_output, anchors):
        assert len(prediction_output.shape) == 3
        num_anchor = anchors.shape[0]
        height, width, channels = prediction_output.shape
        assert channels % num_anchor == 0

        num_class = int(channels / num_anchor) - 5
        assert num_class == len(self.labels)

        outputs = prediction_output.reshape((height, width, num_anchor, -1))

        # Extract bouding box information
        x = (self._logistic(outputs[..., 0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (self._logistic(outputs[..., 1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height

        # (x,y) in the network outputs is the center of the bounding box. Convert them to top-left.
        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, w, h), axis=-1).reshape(-1, 4)

        # Get confidence for the bounding boxes.
        objectness = self._logistic(outputs[..., 4])

        # Get class probabilities for the bounding boxes.
        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness[..., np.newaxis]
        class_probs = class_probs.reshape(-1, num_class)

        assert len(boxes) == len(class_probs)
        return (boxes, class_probs)
    
    def predict_image(self, image):
        #inputs = self.preprocess(image)
        prediction_outputs = self.predict(image)
        return self.postprocess(prediction_outputs)
    
    def preprocess(self, image):
        image = image.convert("RGB") if image.mode != "RGB" else image
        ratio = math.sqrt(self.DEFAULT_INPUT_SIZE / image.width / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        new_width = 32 * round(new_width / 32);
        new_height = 32 * round(new_height / 32);
        image = image.resize((new_width, new_height))
        return image
    
    def predict(self, preprocessed_inputs):
        """Evaluate the model and get the output

        Need to be implemented for each platforms. i.e. TensorFlow, CoreML, etc.
        """
        raise NotImplementedError
    
    def postprocess(self, prediction_outputs):
        """ Extract bounding boxes from the model outputs.

        Args:
            prediction_outputs: Output from the object detection model. (H x W x C)

        Returns:
            List of Prediction objects.
        """
        boxes, class_probs = self._extract_bb(prediction_outputs, self.ANCHORS)

        # Remove bounding boxes whose confidence is lower than the threshold.
        max_probs = np.amax(class_probs, axis=1)
        index, = np.where(max_probs > self.prob_threshold)
        index = index[(-max_probs[index]).argsort()]

        # Remove overlapping bounding boxes
        selected_boxes, selected_classes, selected_probs = self._non_maximum_suppression(boxes[index],
                                                                                         class_probs[index],
                                                                                         self.max_detections)

        listaRetorno = []
        for i in range(len(selected_boxes)):
            
            listaIdentificadas = []
       
            listaIdentificadas.append(round(float(selected_probs[i]), 8))
            listaIdentificadas.append(self.labels[selected_classes[i]])
            listaIdentificadas.append(round(float(selected_boxes[i][0]), 8))
            listaIdentificadas.append(round(float(selected_boxes[i][1]), 8))
            listaIdentificadas.append(round(float(selected_boxes[i][2]), 8))
            listaIdentificadas.append(round(float(selected_boxes[i][3]), 8))
            
            listaRetorno.append(listaIdentificadas)
      
        return listaRetorno

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""
    
    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")
    
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]


def main(image_filename):
    # Load a TensorFlow model
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)
    
    predictions = od_model.predict_image(image_filename)
    return predictions


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        #print('USAGE: {} image_filename'.format(sys.argv[0]))
        
        #coordenadas = main(IMAGE_FILENAME)
        
        
        color = (255, 255, 0)
        
        
        thickness = 4
        captura = cv2.VideoCapture(0)
         
        while(1):
            ret, frame = captura.read()
            coordenadas = main(frame)
            DEFAULT_INPUT_SIZE = 512 * 512
            ratio = math.sqrt(DEFAULT_INPUT_SIZE / frame.shape[1] / frame.shape[0])
            new_width = int(frame.shape[1]* ratio)
            new_height = int(frame.shape[0] * ratio)
            new_width = 32 * round(new_width / 32);
            new_height = 32 * round(new_height / 32);
            dim = (new_width, new_height)
            color = (255, 255, 0)
            for i in range(len(coordenadas)):
                for j in range(len(coordenadas[i])):
                
                    start_point = ( int(new_width*coordenadas[i][2]), int(new_height*coordenadas[i][3]))  
                    end_point = (int((coordenadas[i][2]+coordenadas[i][4])*new_width),int((coordenadas[i][3]+coordenadas[i][5])*new_height)) 
                    retIdentificador = cv2.rectangle(frame, start_point, end_point, color, thickness) 
                    
                    start_point = (int(new_width*coordenadas[i][2])-2, int(new_height*coordenadas[i][3])) 
                    end_point = (int(new_width*coordenadas[i][2])+150, int(new_height*coordenadas[i][3])-25)  
                    retTag = cv2.rectangle(retIdentificador, start_point, end_point, color, -1) 
                    
                    cv2.putText(retTag,""+coordenadas[i][1]+":"+str(int(coordenadas[i][0]*100))+"%", (int(new_width*coordenadas[i][2])+2,int(new_height*coordenadas[i][3])-7), cv2.FONT_ITALIC, 0.5, 0)

            cv2.imshow("Video", frame)
            # grab the current frame
            frame = cv2.resize(frame, (640, 480))  # resize the frame
            encoded, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer)
            footage_socket.send(jpg_as_text)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
         
        captura.release()
        cv2.destroyAllWindows()
        
    else:
        main(sys.argv[1])
