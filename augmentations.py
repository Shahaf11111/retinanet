import albumentations as A
import albumentations.augmentations.transforms as T
import cv2
import matplotlib.pyplot as plt

from retinanet.dataloader import CSVDataset


def gray(p=0.5):
    return T.ToGray(p)


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def transform_img(image_path, bboxes, transform_list, visualize_image=False, format='coco', category_id_to_name={0: 'wheat'}):
    bboxes = get_as_coco_bboxes(bboxes)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose(transform_list, bbox_params=A.BboxParams(format=format, label_fields=['category_ids']))
    transformed = transform(image=image, bboxes=bboxes, category_ids=[0 for v in bboxes])
    if visualize_image:
        visualize(
            transformed['image'],
            transformed['bboxes'],
            transformed['category_ids'],
            category_id_to_name,
        )
    else:
        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    

def get_as_coco_bboxes(bboxes):
    coco_boxes = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        coco_boxes.append([x1, y1, x2-x1, y2-y1])
    return coco_boxes


def main():
    augs = [
        A.OneOf([
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2)]),
        A.OneOf([
            A.ToGray(p=0.2),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                         blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.08, always_apply=False, p=0.2),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6,
                             num_flare_circles_upper=10, src_radius=100, src_color=(255, 255, 255), always_apply=False,
                             p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5,
            always_apply=False, p=0.2)])
     ]
    # a, b, c = transform_img(img_p, boxes, basic_transforms + augs)
    dataset_train = CSVDataset(train_file='configurations/train.csv',
                               class_list='configurations/classes.csv',
                               transform=augs)
    for x in dataset_train:
        img, annot = x['img'], x['annot']
        visualize_annot(img, annot)
        plt.show()
        break
    # print(b)
    # print(c)


def visualize_annot(image, annot, category_id_to_name=None):
    if category_id_to_name is None:
        category_id_to_name = {0: 'wheat'}
    img = image.copy()
    for x1, y1, x2, y2, category_id in annot:
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, [x1, y1, x2 - x1, y2 - y1], class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)



def create_transforms():
    pass
#     ToGray(p=1)
# HorizontalFlip(p=1)
# VerticalFlip(p=1)
# A.Blur(blur_limit=(15, 15), p=1)
# RandomCrop(height, width, p=1)
# A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)
# RandomBrightness(limit=0.2, always_apply=False, p=0.5)
# RandomContrast(limit=0.2, always_apply=False, p=0.5)
# ToGray(always_apply=False, p=0.5)
# Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5)
# RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5)
# RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5)
# RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5)
# RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5)
# RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5)


if __name__ == "__main__":
    main()