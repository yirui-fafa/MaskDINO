
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image
import cv2
import os
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as Fv
import time
import numpy as np
import tqdm

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "traffic_train": (
        "ori_images",
        "labels",
        "instance_masks",
        "ImageSets/Main/train_json.txt"
    ),
    "traffic_val": (
        "ori_images",
        "labels",
        "instance_masks",
        "ImageSets/Main/val_json.txt"
    ),
}

TRAFFIC_CATEGORIES = [{'id': 1, 'name': 'person'},
                      {'id': 2, 'name': 'non-motor'},
                      {'id': 3, 'name': 'car'},
                      {'id': 4, 'name': 'tricycle'}]

def _get_traffic_instances_meta():
    thing_ids = [k["id"] for k in TRAFFIC_CATEGORIES]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in TRAFFIC_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret
     

def suprevison_dataset_function(image_root,
                                label_json_root,
                                instance_mask_root,
                                image_list_txt,
                                metadata):
    dataset_dicts = []
    with open(image_list_txt, 'r') as fp:
            source_files = fp.readlines()
    # import pdb; pdb.set_trace()
    
    # ids = 0
    # rm_ids = 0
    for image_file in source_files:
        # ids+=1
        # print(ids)
        image_file_name = os.path.basename(image_file).strip()
        record = {}
        record["file_name"] = os.path.join(image_root, image_file_name)
        label_file_name = image_file_name.replace('.png', '.json').replace('.jpg', '.json')
        label_file = os.path.join(label_json_root, label_file_name)
        # t0 = time.time()
        with open(label_file, 'r') as fr:
            json_data = json.load(fr)
        # print("open_label_file:", time.time()-t0)
        record["height"] = json_data["height"]
        record["width"] = json_data["width"]
        height = json_data["height"]
        width = json_data["width"]
        # record["image_id"] = json_data["id"]

        segs_info = json_data['segs']["instance_mask_path"]
        labels = json_data['labels'] # num_obj
        boxes_list = json_data['boxes']['boxes'] # num_obj, 4  （cx,cy,w,h 相对坐标）
        obj_ids = json_data['ids']
        # rm unlabeld img filter empty
        if len(obj_ids) == 0:
            # rm_ids += 1
            continue
        record["obj_ids"] = obj_ids
        instance_mask_path = os.path.join(instance_mask_root, segs_info)
        record["instance_mask_path"] = instance_mask_path
        # t1 = time.time()
        # segm_list = convert_mask_to_poly(instance_mask_path, obj_ids) # num_obj, n (n个关键点,xyxy)
        # print("convert_mask_to_poly:", time.time()-t1)
        objs = []
        # t2 = time.time()
        for idx, box in enumerate(boxes_list):
            obj = {}
            obj["iscrowd"] = 0
            obj["bbox"] = [box[0], box[1],  box[2], box[3]]
            # convert x,y,w,h abs
            # obj["bbox"] = [(box[0]-box[2]/2)*width, (box[1]-box[3]/2)*height, box[2]*width, box[3]*height]
            # obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = labels[idx]
            # obj["segmentation"] = segm_list[idx]
            # obj["segmentation"] = instance_mask_path
            objs.append(obj)
        record["annotations"] = objs
        # print("solve_box:", time.time()-t2)
        dataset_dicts.append(record)


    return dataset_dicts

def register_suprevison_dataset_load(name, 
                                    metadata, 
                                    image_root, 
                                    label_json_root, 
                                    instance_mask_root, 
                                    image_list_txt):
    
    DatasetCatalog.register(
        name, 
        lambda: suprevison_dataset_function(
            image_root, label_json_root, instance_mask_root, image_list_txt, metadata
        ),
    )
    
    # later, to access the data:
    # data: List[Dict] = DatasetCatalog.get("suprevison_dataset")
    # 评估方式可以参考评估的写，先空着先 以sam_interactive的评估为例
    # MetadataCatalog.get(name).set(
    #     tsv_file=tsv_file, evaluator_type="sam_interactive",  **metadata
    # )



def register_suprevison_dataset(root):
    
    metadata = _get_traffic_instances_meta()

    for (
        prefix,
        (image_root, label_json_root, instance_mask_root, image_list_txt),
    ) in _PREDEFINED_SPLITS.items():
        register_suprevison_dataset_load(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, label_json_root),
            os.path.join(root, instance_mask_root),
            os.path.join(root, image_list_txt),
        )
   


_root = os.getenv("SUPREVISION_DATASETS", "datasets")
register_suprevison_dataset(_root)


    
