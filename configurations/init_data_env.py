import pandas as pd
import os
import random
import shutil
from ast import literal_eval as str_to_list


def is_dir_empty(dir_name, create_dir=False):
    if not os.path.exists(dir_name):  # Dir does not exist, create a new dir
        if create_dir:
            os.mkdir(dir_name)
            return True
        return False
    if os.path.isdir(dir_name) and not os.listdir(dir_name):
        return True
    return False


def randomly_transfer_files_by_fraction(src_dir, target_dir, fraction):
    """
    src_dir - directory with files in it (only files, no sub-directories!)
    target_dir - empty directory or a path that does not exist
    fraction - of files from 'src_dir' to transfer to 'target_dir'
    """
    if not is_dir_empty(src_dir) and is_dir_empty(target_dir, True):
        src_file_names = os.listdir(src_dir)
        amount_to_transfer = int(len(src_file_names) * fraction)
        transferred = []
        for i in range(amount_to_transfer):
            file_name = random.choice(src_file_names)
            shutil.move(os.path.join(src_dir, file_name),
                        os.path.join(target_dir, file_name))
            transferred.append(file_name)
            src_file_names.remove(file_name)
        print('Transfer is successfull:\n{} directory has {} files.\n'
              '{} directory has {} files.'.format(src_dir, len(os.listdir(src_dir)),
                                                  target_dir, len(os.listdir(target_dir))))
    else:
        print("Failed to transfer files from {} to {}".format(src_dir, target_dir))


def format_bbox(bbox):
    bbox_list = str_to_list(bbox)
    x, y, w, h = bbox_list
    return [int(x), int(y), int(x + w), int(y + h)]


def format_df_bbox(df):
    n_rows = len(df)
    df['x1'] = '0'
    df['y1'] = '0'
    df['x2'] = '0'
    df['y2'] = '0'
    for idx, row in df.iterrows():
        bbox = row['bbox']
        x, y, w, h = str_to_list(bbox)
        df.loc[idx, ['x1', 'x2', 'y1', 'y2']] = [int(x), int(x + w), int(y), int(y + h)]
        if idx % 2500 == 0:
            print('Formatted rows: {}/{}'.format(idx, n_rows))
    return df


def validate_df(df):
    count_x, count_y, count_zero = 0, 0, 0
    for i, row in df.iterrows():
        if int(row['x1']) > int(row['x2']):
            count_x += 1
        if int(row['y1']) > int(row['y2']):
            count_y += 1
        if int(row['x1']) == 0 and  int(row['x2']) == 0 and int(row['y2']) == 0 and int(row['y2']) == 0:
            count_zero += 1
    print("x:{}, y:{}, zero:{}".format(count_x, count_y, count_zero))


def format_df(df, _dir):
    i = 0
    img_name_list = os.listdir(_dir)
    n_names = len(img_name_list)
    for img_name in img_name_list:
        img_id = img_name.replace('.jpg', '')
        df.loc[df['image_id'] == img_id, 'image_id'] = os.path.join(_dir, img_name)
        i += 1
        if i % 250 == 0:
            print('Formatted images: {}/{}'.format(i, n_names))
    df = df.loc[df['image_id'].str.contains('C:')]
    df['class_name'] = 'wheat'
    df.drop(columns=['width', 'height', 'source', 'bbox'], inplace=True)
    return df[['image_id', 'x1', 'y1', 'x2', 'y2', 'class_name']]


if __name__ == '__main__':
    csv_path = 'C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\GlobalWheatDetectionData\\train.csv'
    train_path = 'C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\GlobalWheatDetectionData\\images\\train'
    valid_path = 'C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\GlobalWheatDetectionData\\images\\valid'

    # randomly_transfer_files_by_fraction(train_path, valid_path, 0.15)

    # df = pd.read_csv(csv_path)
    # df = format_df_bbox(df)
    # df.to_csv('C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\configurations\\all.csv', index=False)

    df = pd.read_csv('C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\configurations\\all.csv')

    train_df = format_df(df, train_path)
    train_df.to_csv('C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\configurations\\train.csv', index=False)

    valid_df = format_df(df, valid_path)
    valid_df.to_csv('C:\\Users\\lenas\\PycharmProjects\\pytorch-retinanet\\configurations\\valid.csv', index=False)

