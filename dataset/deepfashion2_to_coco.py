# https://github.com/switchablenorms/DeepFashion2/blob/master/evaluation/deepfashion2_to_coco.py

import json
from PIL import Image
import numpy as np
import argparse
import os

dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

dataset['categories'].append({
    'id': 1,
    'name': "short_sleeved_shirt",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 2,
    'name': "long_sleeved_shirt",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 3,
    'name': "short_sleeved_outwear",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 4,
    'name': "long_sleeved_outwear",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 5,
    'name': "vest",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 6,
    'name': "sling",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 7,
    'name': "shorts",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 8,
    'name': "trousers",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 9,
    'name': "skirt",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 10,
    'name': "short_sleeved_dress",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 11,
    'name': "long_sleeved_dress",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 12,
    'name': "vest_dress",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})
dataset['categories'].append({
    'id': 13,
    'name': "sling_dress",
    'supercategory': "clothes",
    'keypoints': [f'{i}' for i in range(1, 295)],
    'skeleton': []
})

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_dir', default='./train/annos/')
parser.add_argument('--image_dir', default='./train/image/')
parser.add_argument('--out_dir', default='./detectron_annos/train_annos.json')
parser.add_argument('--num_images', type=int, default=10)
args = parser.parse_args()

sub_index = 0  # the index of ground truth instance
for num in range(1, args.num_images + 1):
    json_name = os.path.join(args.annotation_dir, str(num).zfill(6) + '.json')
    image_name = os.path.join(args.image_dir, str(num).zfill(6) + '.jpg')

    if num >= 0:
        imag = Image.open(image_name)
        width, height = imag.size
        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
            pair_id = temp['pair_id']

            dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': str(num).zfill(6) + '.jpg',
                'flickr_url': '',
                'id': num,
                'license': 0,
                'width': width,
                'height': height
            })
            for i in temp:
                if i == 'source' or i == 'pair_id':
                    continue
                else:
                    points = np.zeros(294 * 3)
                    sub_index = sub_index + 1
                    box = temp[i]['bounding_box']
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    x_1 = box[0]
                    y_1 = box[1]
                    bbox = [x_1, y_1, w, h]
                    cat = temp[i]['category_id']
                    style = temp[i]['style']
                    seg = temp[i]['segmentation']
                    landmarks = temp[i]['landmarks']

                    points_x = landmarks[0::3]
                    points_y = landmarks[1::3]
                    points_v = landmarks[2::3]
                    points_x = np.array(points_x)
                    points_y = np.array(points_y)
                    points_v = np.array(points_v)

                    if cat == 1:
                        for n in range(0, 25):
                            points[3 * n] = points_x[n]
                            points[3 * n + 1] = points_y[n]
                            points[3 * n + 2] = points_v[n]
                    elif cat == 2:
                        for n in range(25, 58):
                            points[3 * n] = points_x[n - 25]
                            points[3 * n + 1] = points_y[n - 25]
                            points[3 * n + 2] = points_v[n - 25]
                    elif cat == 3:
                        for n in range(58, 89):
                            points[3 * n] = points_x[n - 58]
                            points[3 * n + 1] = points_y[n - 58]
                            points[3 * n + 2] = points_v[n - 58]
                    elif cat == 4:
                        for n in range(89, 128):
                            points[3 * n] = points_x[n - 89]
                            points[3 * n + 1] = points_y[n - 89]
                            points[3 * n + 2] = points_v[n - 89]
                    elif cat == 5:
                        for n in range(128, 143):
                            points[3 * n] = points_x[n - 128]
                            points[3 * n + 1] = points_y[n - 128]
                            points[3 * n + 2] = points_v[n - 128]
                    elif cat == 6:
                        for n in range(143, 158):
                            points[3 * n] = points_x[n - 143]
                            points[3 * n + 1] = points_y[n - 143]
                            points[3 * n + 2] = points_v[n - 143]
                    elif cat == 7:
                        for n in range(158, 168):
                            points[3 * n] = points_x[n - 158]
                            points[3 * n + 1] = points_y[n - 158]
                            points[3 * n + 2] = points_v[n - 158]
                    elif cat == 8:
                        for n in range(168, 182):
                            points[3 * n] = points_x[n - 168]
                            points[3 * n + 1] = points_y[n - 168]
                            points[3 * n + 2] = points_v[n - 168]
                    elif cat == 9:
                        for n in range(182, 190):
                            points[3 * n] = points_x[n - 182]
                            points[3 * n + 1] = points_y[n - 182]
                            points[3 * n + 2] = points_v[n - 182]
                    elif cat == 10:
                        for n in range(190, 219):
                            points[3 * n] = points_x[n - 190]
                            points[3 * n + 1] = points_y[n - 190]
                            points[3 * n + 2] = points_v[n - 190]
                    elif cat == 11:
                        for n in range(219, 256):
                            points[3 * n] = points_x[n - 219]
                            points[3 * n + 1] = points_y[n - 219]
                            points[3 * n + 2] = points_v[n - 219]
                    elif cat == 12:
                        for n in range(256, 275):
                            points[3 * n] = points_x[n - 256]
                            points[3 * n + 1] = points_y[n - 256]
                            points[3 * n + 2] = points_v[n - 256]
                    elif cat == 13:
                        for n in range(275, 294):
                            points[3 * n] = points_x[n - 275]
                            points[3 * n + 1] = points_y[n - 275]
                            points[3 * n + 2] = points_v[n - 275]
                    num_points = len(np.where(points_v > 0)[0])

                    dataset['annotations'].append({
                        'area': w * h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'pair_id': pair_id,
                        'image_id': num,
                        'iscrowd': 0,
                        'style': style,
                        'num_keypoints': num_points,
                        'keypoints': points.tolist(),
                        'segmentation': seg,
                    })

with open(args.out_dir, 'w') as f:
    json.dump(dataset, f)
