#!/usr/bin/env python3
"""A YOLO objection detection model class"""


import tensorflow.keras as K
import numpy as np
import cv2
import os


class Yolo:
    """A YOLO object detection class """
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path) as class_file:
            self.class_names = [line.rstrip('\n') for line in class_file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ Process Darknet outputs """
        boxes = [output[:, :, :, 0:4] for output in outputs]
        for oidx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    centery = ((1 / (1 + np.exp(-output[y, x, :, 1])) + y)
                               / output.shape[0] * image_size[0])
                    centerx = ((1 / (1 + np.exp(-output[y, x, :, 0])) + x)
                               / output.shape[1] * image_size[1])
                    prior_resizes = self.anchors[oidx].astype(float)
                    prior_resizes[:, 0] *= (np.exp(output[y, x, :, 2])
                                            / 2 * image_size[1] /
                                            self.model.input.shape[1].value)
                    prior_resizes[:, 1] *= (np.exp(output[y, x, :, 3])
                                            / 2 * image_size[0] /
                                            self.model.input.shape[2].value)
                    output[y, x, :, 0] = centerx - prior_resizes[:, 0]
                    output[y, x, :, 1] = centery - prior_resizes[:, 1]
                    output[y, x, :, 2] = centerx + prior_resizes[:, 0]
                    output[y, x, :, 3] = centery + prior_resizes[:, 1]
        box_confidences = [1 / (1 + np.exp(-output[:, :, :, 4, np.newaxis]))
                           for output in outputs]
        box_class_probs = [1 / (1 + np.exp(-output[:, :, :, 5:]))
                           for output in outputs]
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter box outputs """
        all_boxes = np.concatenate([boxs.reshape(-1, 4) for boxs in boxes])
        class_probs = np.concatenate([probs.reshape(-1,
                                                    box_class_probs[0].
                                                    shape[-1])
                                      for probs in box_class_probs])
        all_classes = class_probs.argmax(axis=1)
        all_confidences = (np.concatenate([conf.reshape(-1)
                                           for conf in box_confidences])
                           * class_probs.max(axis=1))
        thresh_idxs = np.where(all_confidences < self.class_t)
        return (np.delete(all_boxes, thresh_idxs, axis=0),
                np.delete(all_classes, thresh_idxs),
                np.delete(all_confidences, thresh_idxs))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ non max suppression on boxes """
        sort_order = np.lexsort((-box_scores, box_classes))
        box_scores = box_scores[sort_order]
        box_classes = box_classes[sort_order]
        filtered_boxes = filtered_boxes[sort_order]
        del_idxs = []
        for idx in range(len(box_scores)):
            if idx in del_idxs:
                continue
            clas = box_classes[idx]
            box = filtered_boxes[idx]
            for cidx in range(idx + 1, len(box_scores)):
                if (box_classes[cidx] != clas):
                    break
                if ((intersection_over_union(filtered_boxes[cidx], box)
                     >= self.nms_t)):
                    del_idxs.append(cidx)
        return (np.delete(filtered_boxes, del_idxs, axis=0),
                np.delete(box_classes, del_idxs),
                np.delete(box_scores, del_idxs))

    @staticmethod
    def load_images(folder_path):
        """ load images from folder """
        file_list = os.listdir(folder_path)
        images = []
        file_paths = []
        for file in file_list:
            path = folder_path + '/' + file
            images.append(cv2.imread(folder_path + '/' + file))
            file_paths.append(path)
        return images, file_paths

    def preprocess_images(self, images):
        """ resize and rescale images """
        image_sizes = np.zeros((len(images), 2))
        target_height = self.model.input.shape[1].value
        target_width = self.model.input.shape[2].value
        proc_images = np.zeros((len(images), target_height, target_width, 3))
        for idx, image in enumerate(images):
            image_sizes[idx][0] = image.shape[0]
            image_sizes[idx][1] = image.shape[1]
            proc_images[idx] = cv2.resize(image,
                                          (target_height, target_width),
                                          interpolation=cv2.INTER_CUBIC)
            proc_images[idx] -= proc_images[idx].min()
            proc_images[idx] /= proc_images[idx].max()
        return proc_images, image_sizes.astype(int)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """ draw boxes on image for prediction """
        image = image.copy()
        for idx, box in enumerate(boxes.astype(int)):
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                                  (255, 0, 0), 1, cv2.LINE_AA)
            class_prob = (self.class_names[box_classes[idx]] +
                          ' ' + "{0:.2f}".format(box_scores[idx]))
            image = cv2.putText(image, class_prob, (box[0], box[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        cv2.imshow(file_name, image)
        if (cv2.waitKey(0) == 115):
            cv2.imwrite('detections/' + file_name, image)
        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """ predict all images in folder """
        originals, file_names = self.load_images(folder_path)
        images, origin_sizes = self.preprocess_images(originals)
        predicts = self.model.predict(images)
        for idx in range(predicts[0].shape[0]):
            outputs = [predict[idx, ...] for predict in predicts]
            processed = self.process_outputs(outputs, origin_sizes[idx])
            threshed = self.filter_boxes(*processed)
            non_maxed = self.non_max_suppression(*threshed)
            self.show_boxes(originals[idx], *non_maxed, file_names[idx])
        outnames = ['detections/' + os.path.split(file)[1]
                    for file in file_names]
        return non_maxed, outnames


def intersection_over_union(boxa, boxb):
    """ calculate intersection over union """
    intx1 = max(boxa[0], boxb[0])
    inty1 = max(boxa[1], boxb[1])
    intx2 = min(boxa[2], boxb[2])
    inty2 = min(boxa[3], boxb[3])

    intarea = max(0, (intx2 - intx1)) * max(0, (inty2 - inty1))
    boxaarea = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1])
    boxbarea = (boxb[2] - boxb[0]) * (boxb[3] - boxb[1])
    return intarea / (boxaarea + boxbarea - intarea)
