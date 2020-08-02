import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import pandas as pd
import re
import retinanet.model as model


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list, visualize):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = load_model(model_path, num_classes=len(classes))
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    all_box_scores = dict()

    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)
            img_box_scores = []

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                if visualize:
                    label_name = labels[int(classification[idxs[0][j]])]
                    print(bbox, classification.shape)
                    score = scores[j]
                    caption = '{} {:.3f}'.format(label_name, score)
                    # draw_caption(img, (x1, y1, x2, y2), label_name)
                    draw_caption(image_orig, (x1, y1, x2, y2), caption)
                    cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                else:
                    img_box_scores.append([float('{:1.5f}'.format(scores[j].item())), x1, y1, x2 - x1, y2 - y1])
            if visualize:
                cv2.imshow('detections', image_orig)
                cv2.waitKey(0)
            else:
                all_box_scores[img_name.split('.')[0]] = img_box_scores
    return all_box_scores


def dict_to_df(d):
    df = pd.DataFrame(columns=['image_id', 'PredictionString'])
    for (img_id, img_preds) in d.items():
        img_preds_str = ' '.join([' '.join([str(p) for p in pred]) for pred in img_preds])
        df.loc[len(df)] = [img_id, img_preds_str]
    return df


def load_model(model_path, num_classes=1):
    depth = int(re.findall(r'\d+', os.path.basename(model_path))[0])
    if depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=False)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=False)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=False)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=False)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    retinanet.load_state_dict(torch.load(model_path)['model'], strict=False)
    return retinanet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument('--visualize', help='True: Visualize the results. False: Creates results.csv file.')

    parser = parser.parse_args()
    results_path = os.path.join(os.path.abspath(os.path.join(parser.model_path, os.pardir)),
                                os.path.basename(parser.model_path).split('.')[0] + '.csv')

    results_dict = detect_image(parser.image_dir, parser.model_path, parser.class_list, parser.visualize)
    if not parser.visualize:
        results_df = dict_to_df(results_dict)
        results_df.to_csv(results_path, index=False)

