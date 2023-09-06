
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

def convert_mask_to_poly(instance_mask_path,
                        obj_ids):
    # t0 = time.time()
    # gt_instance_masks = Image.open(instance_mask_path)
    
    t1 = time.time()
    # print("pil_read:", t1-t0)
    gt_instance_masks = cv2.imread(instance_mask_path, cv2.IMREAD_GRAYSCALE)
    # t2 = time.time()
    # print("cv_read:", t2-t1)
    gt_instance_masks = torch.tensor(gt_instance_masks, dtype=torch.uint8).cuda()
    # t21 = time.time()
    # print("cvmask_to_tensor:", t21-t2)
    
    obj_ids = torch.tensor(obj_ids, dtype=torch.long)
    # 因为分割训练空图需要过滤掉，这个不需要了
    # if obj_ids.shape[0] == 0: 
    #     return [[]]
    max_obj_id = obj_ids.max()
    
    # resize small
    ori_h = gt_instance_masks.shape[0]
    ori_w = gt_instance_masks.shape[1]
    # resize_h = gt_instance_masks.shape[0]//4
    # resize_w = gt_instance_masks.shape[1]//4
    resize_h = 270
    resize_w = 480
    raito_h = ori_h/resize_h
    ratio_w = ori_w/resize_w
    gt_instance_masks = Fv.resize(gt_instance_masks.clone().unsqueeze(0), [resize_h, resize_w], interpolation=InterpolationMode.NEAREST).squeeze(0)

    binary_instance_mask = F.one_hot(gt_instance_masks.long(), num_classes=max_obj_id + 1).permute(2, 0, 1)  # max_obj_id+1, pad_H, pad_W
    # t22 = time.time()
    # print("tensormask_to_onehot:", t22-t21)
    
    binary_instance_mask = binary_instance_mask[obj_ids.long(), :, :].to(torch.uint8).cpu() # num_obj, pad_H, pad_W
    # t23 = time.time()
    # print("onehotmask_to_obj:", t23-t22)

    
    # resize mask ori
    # binary_instance_mask = Fv.resize(binary_instance_mask.clone(), [ori_h, ori_w], interpolation=InterpolationMode.NEAREST).to(torch.uint8)
    # t233 = time.time()
    # print("unresize:", t233-t23)

    binary_instance_mask = binary_instance_mask.detach().numpy()
    torch.cuda.empty_cache()
    # t3 = time.time()
    # print("objmask_to_mumpy:", t3-t233)
    # print("mask_to_numpy:", t3-t2)
    contours = []
    for mask in binary_instance_mask:
        # mask = cv2.UMat(mask)
        contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1) #CHAIN_APPROX_TC89_L1
        # import pdb; pdb.set_trace()
        reshaped_contour = []
        for entity in contour:
            assert len(entity.shape) == 3
            assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
            # reshaped_contour.append(entity.reshape(-1).tolist())
            reshapeed_entity = np.array([[ent[0]*ratio_w, ent[1]*raito_h] for ent in entity.squeeze(1)])  # [x,y]
            reshaped_contour.append(reshapeed_entity.reshape(-1).tolist())
        contours.append(reshaped_contour)
    # t4 = time.time()
    # print("numpymask_to_ploy:", t4-t3)
    
    
    # 测试=====
    # resize_h = gt_instance_masks.shape[0]//4
    # resize_w = gt_instance_masks.shape[1]//4
    # resized_masks = Fv.resize(gt_instance_masks.clone().unsqueeze(0), [resize_h, resize_w], interpolation=InterpolationMode.NEAREST).squeeze(0)
    # t5 = time.time()
    # print("resize_mask:", t5-t4)
    # re_binary_instance_mask = F.one_hot(resized_masks.long(), num_classes=max_obj_id + 1).permute(2, 0, 1)  # max_obj_id+1, pad_H, pad_W
    # t62 = time.time()
    # print("tensormask_to_onehot:", t62-t5)
    
    # re_binary_instance_mask = re_binary_instance_mask[obj_ids.long(), :, :].to(torch.uint8).cpu() # num_obj, pad_H, pad_W
    # t63 = time.time()
    # print("onehotmask_to_obj:", t63-t62)
    
    # re_binary_instance_mask = re_binary_instance_mask.detach().numpy()
    # t7 = time.time()
    # print("objmask_to_mumpy:", t7-t63)
    # # print("mask_to_numpy:", t3-t2)
    # re_contours = []
    # for mask in re_binary_instance_mask:
    #     # mask = cv2.UMat(mask)
    #     contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #CHAIN_APPROX_TC89_L1
    #     # import pdb; pdb.set_trace()
    #     reshaped_contour = []
    #     for entity in contour:
    #         assert len(entity.shape) == 3
    #         assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
    #         reshaped_contour.append(entity.reshape(-1).tolist())
    #     re_contours.append(reshaped_contour)
    # t8 = time.time()
    # print("numpymask_to_ploy:", t8-t7)
    
    
    #=========

    # import pdb; pdb.set_trace()
    return contours        

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
            # convert x,y,w,h abs
            obj["bbox"] = [(box[0]-box[2]/2)*width, (box[1]-box[3]/2)*height, box[2]*width, box[3]*height]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
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


    
