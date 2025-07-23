import os
import numpy as np
import cv2
import json
import glob
import torch
import math
import onnxruntime
import argparse
from pycocotools import mask as mask_utils
from skimage import measure
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_erosion


def resize_image(file_name, in_h = 1024, in_w = 1024):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    if width < in_w and height < in_h:
        border_type = cv2.BORDER_CONSTANT
        border_color = [0, 0, 0]
        padded = cv2.copyMakeBorder(
            img,
            0, 
            in_h - height, 
            0, 
            in_w - width,
            borderType=border_type,
            value=border_color
        )
        return padded, 1.0
    else:
        w_ratio = in_w / width
        h_ratio = in_h / height
        ratio = w_ratio if w_ratio < h_ratio else h_ratio
        resized_img = cv2.resize(
            img,
            None,
            fx=ratio,
            fy=ratio,
            interpolation=cv2.INTER_CUBIC
        )
        resized_height, resized_width = resized_img.shape[:2]
        border_type = cv2.BORDER_CONSTANT
        border_color = [0, 0, 0]
        padded = cv2.copyMakeBorder(
            resized_img,
            0,
            in_h - resized_height, 
            0, 
            in_w - resized_width,
            borderType=border_type,
            value=border_color
        )
        return padded, ratio 


def normalize_image(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img / 255.0 - mean) / std
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :].astype(np.float32)
    return img


def resize_points(points, ratio):
    return [[point[0] * ratio, point[1] * ratio] for point in points]


def preprocess_image(file_name, points, in_h = 1024, in_w = 1024):
    img, ratio = resize_image(file_name, in_h, in_w)
    img = normalize_image(img)
    return img, ratio


def get_all_json_file(path):
    return glob.glob(f"{path}/*.json")


def get_item(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data["image"]["file_name"], data["image"]["width"], data["image"]["height"], data["annotations"]


def create_encoder_decoder(args):
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

    encoder_path = args.enc_onnx
    encoder_session = onnxruntime.InferenceSession(encoder_path, providers=providers)

    decoder_path = args.dec_onnx
    decoder_session = onnxruntime.InferenceSession(decoder_path, providers=providers)
    return encoder_session, decoder_session


def encoder_inference(encoder, img):
    encoder_input_names = [ele.name for ele in encoder.get_inputs()]
    encoder_output_names = [ele.name for ele in encoder.get_outputs()]

    image_embeddings, high_res_features_1, high_res_features_2 = encoder.run(
        encoder_output_names, {encoder_input_names[0]: img}
    )
    return image_embeddings, high_res_features_1, high_res_features_2


def decoder_inference(decoder, points, img, image_embedings, high_res_features_1, high_res_features_2, idx=0, obj_idx=0, args=None):
    assert len(points) <= 8

    img_h, img_w = img.shape[2:]
    # print(img_h, img_w, img.shape)
    mask_input = np.zeros(
        (
            1,
            1,
            img_h // 4,
            img_w // 4,
        ),
        dtype=np.float32,
    )
    has_mask_input = np.array([0], dtype=np.float32)
    decoder_input_names = [ele.name for ele in decoder.get_inputs()]
    decoder_output_names = [ele.name for ele in decoder.get_outputs()]

    labels = [1] * len(points)
    points.extend([[0, 0]] * (8 - len(points)))
    labels.extend([-1] * (8 - len(labels)))
    points, labels = np.array(points).astype(np.float32), np.array(labels).astype(np.float32)
    points = points[np.newaxis, ...]
    labels = labels[np.newaxis, ...]
    # print(points)
    # print(labels)
    # print(points.shape)
    # print(labels.shape)
    # print(mask_input.shape)
    ori_size = np.array(img.shape[2:], dtype=np.int64)

    if args.dump_tensor:
        image_embed_tensor = torch.from_numpy(image_embedings)
        high_res_features_1_tensor = torch.from_numpy(high_res_features_1)
        high_res_features_2_tensor = torch.from_numpy(high_res_features_2)
        points_tensor = torch.from_numpy(points)
        labels_tensor = torch.from_numpy(labels)
        mask_input_tensor = torch.from_numpy(mask_input)
        has_mask_input_tensor = torch.from_numpy(has_mask_input)
        ori_size_tensor = torch.from_numpy(ori_size)

        torch.save(image_embed_tensor, os.path.join(args.dump_dir, f"image_embed_tensor_{idx}_{obj_idx}.pt"))
        torch.save(high_res_features_1_tensor, os.path.join(args.dump_dir, f"high_res_features_1_tensor_{idx}_{obj_idx}.pt"))
        torch.save(high_res_features_2_tensor, os.path.join(args.dump_dir, f"high_res_features_2_tensor_{idx}_{obj_idx}.pt"))
        torch.save(points_tensor, os.path.join(args.dump_dir, f"point_tensor_{idx}_{obj_idx}.pt"))
        torch.save(labels_tensor, os.path.join(args.dump_dir, f"labels_tensor_{idx}_{obj_idx}.pt"))
        torch.save(mask_input_tensor, os.path.join(args.dump_dir, f"mask_input_tensor_{idx}_{obj_idx}.pt"))
        torch.save(has_mask_input_tensor, os.path.join(args.dump_dir, f"has_mask_input_tensor_{idx}_{obj_idx}.pt"))
        torch.save(ori_size_tensor, os.path.join(args.dump_dir, f"ori_size_tensor_{idx}_{obj_idx}.pt"))


    inputs = [
        image_embedings, 
        high_res_features_1, 
        high_res_features_2,
        points, 
        labels, 
        mask_input, 
        has_mask_input, 
        ori_size
    ]
    
    outputs = decoder.run(
        decoder_output_names,
        {
            name: input 
            for name, input in zip(decoder_input_names, inputs)
        },
    )
    masks = outputs[0]
    scores = outputs[1]
    low_res_masks = outputs[2]
    max_idx = np.argmax(scores[0])
    mask = masks[0][max_idx]
    mask[mask > 0.0] = 255
    low_res_mask = low_res_masks[0][max_idx]
    low_res_mask = np.array([[low_res_mask]])
    return mask, low_res_mask


def resize_back_infer_mask(infer_mask, ratio, ori_h, ori_w):
    resize_mask = cv2.resize(
        infer_mask,
        None,
        fx=1 / ratio,
        fy=1 / ratio,
        interpolation=cv2.INTER_CUBIC
    )
    return resize_mask[:ori_h, :ori_w]


def jaccard(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 1.0


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    foreground_mask[foreground_mask > 0] = 1
    foreground_mask[foreground_mask < 0] = 0
    gt_mask[gt_mask > 0] = 1
    gt_mask[gt_mask < 0] = 0

    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    annotation[annotation > 0] = 1
    annotation[annotation < 0] = 0
    segmentation[segmentation > 0] = 1
    segmentation[segmentation < 0] = 0

    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def benchmark(args):

    encoder, decoder = create_encoder_decoder(args)

    j_f_score_sum = 0
    j_f_obj_num = 0
    miou_sum = 0
    json_list = get_all_json_file(args.data_dir)
    j = 0
    for json_file in json_list:
        j = j + 1
        file_name, img_w, img_h, annotations = get_item(json_file)
        img, ratio = resize_image(os.path.join(args.data_dir, file_name))
        img = normalize_image(img)

        img_tensor = torch.from_numpy(img)
        if args.dump_tensor:
            torch.save(img_tensor, os.path.join(args.dump_dir, f"img_{j}.pt"))
        image_embeddings, high_res_features_1, high_res_features_2 = encoder_inference(encoder, img)

        i = 0
        for anno in annotations:
            i = i + 1
            mask, points = mask_utils.decode(anno["segmentation"]), anno["point_coords"]
            mask[mask > 0] = 255
            points = resize_points(points, ratio)
            infer_mask, _ = decoder_inference(decoder, points, img, image_embeddings, high_res_features_1, high_res_features_2, j, i, args.dump_tensor)
            infer_mask = resize_back_infer_mask(infer_mask, ratio, img_h, img_w)
            j_score = db_eval_iou(mask, infer_mask)
            f_score = f_measure(mask, infer_mask)
            # print(mask.shape, infer_mask.shape)
            j_f_score_sum += (j_score + f_score) / 2
            miou_sum += j_score
            j_f_obj_num += 1
            j_f_score = j_f_score_sum / j_f_obj_num
            miou = miou_sum / j_f_obj_num
            print(f"file_name:{file_name}, image_num:{j} obj_num:{i}, cur_j:{j_score} cur_f:{f_score} avg miou:{miou} avg j&f:{j_f_score}")


def main():
    parser = argparse.ArgumentParser(description="export sam2 onnx")
    parser.add_argument("--data_dir", type=str, help="path")
    parser.add_argument("--enc_onnx", type=str, help="path")
    parser.add_argument("--dec_onnx", type=str, help="path")
    parser.add_argument("--dump_tensor", action='store_true', help="dump tensor enable")
    parser.add_argument("--dump_dir", type=str, help="dump tensor path")
    args = parser.parse_args()

    benchmark(args)


if __name__ == "__main__":
    main()