import os
import time
import argparse
import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn
from typing import Any
from onnxsim import simplify
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @torch.no_grad()
    def forward(
        self, 
        input: torch.Tensor
    ):
        backbone_out = self.model.forward_image(input)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings = feats[2]
        high_res_features1 = feats[0]
        high_res_features2 = feats[1]
        return image_embeddings, high_res_features1, high_res_features2

class ImageDecoder(nn.Module):

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor, # [1, 32, 256, 256]
        high_res_features2: torch.Tensor, # [1, 64, 128, 128]
        point_coords: torch.Tensor, # [num_labels,num_points,2]
        point_labels: torch.Tensor, # [num_labels,num_points]
        mask_input: torch.Tensor,  # [1,1,256,256]
        has_mask_input: torch.Tensor,  # [1]
        orig_im_size: torch.Tensor   # [2]
    ):
        sparse_embedding = self.embed_points(point_coords, point_labels)
        dense_embedding = self.embed_masks(mask_input, has_mask_input)
        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        low_res_masks = torch.clamp(masks, -32.0, 32.0)
        masks = torch.nn.functional.interpolate(
            masks,
            size=(orig_im_size[0], orig_im_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks, iou_predictions, low_res_masks

    def embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:

        point_coords = point_coords + 0.5

        padding_point = torch.zeros(
            (point_coords.shape[0], 1, 2), device=point_coords.device
        )
        padding_label = -torch.ones(
            (point_labels.shape[0], 1), device=point_labels.device
        )
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(
            point_coords
        )
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.prompt_encoder.not_a_point_embed.weight
            * (point_labels == -1)
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding


def export_image_encoder(model, onnx_path):
    onnx_path = os.path.join(onnx_path, "image_encoder.onnx")
    input_img = torch.randn(1, 3,1024, 1024)
    output_names = ["image_embeddings", "high_res_features1", "high_res_features2"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
    )


def export_image_decoder(model, onnx_path):
    onnx_path = os.path.join(onnx_path, "image_decoder.onnx")
    image_embeddings = torch.randn(1,256,64,64)
    high_res_features1 = torch.randn(1,32,256,256)
    high_res_features2 = torch.randn(1,64,128,128)
    point_coords = torch.randn(1,8,2)
    point_labels = torch.randn(1,8)
    mask_input = torch.randn(1, 1, 256, 256, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    orig_im_size = torch.tensor([1024,1024],dtype=torch.int64)
    input_name = [
        "image_embeddings",
        "high_res_features1",
        "high_res_features2",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
        "orig_im_size"
    ]
    output_name = ["masks", "iou_predictions", "low_res_masks"]
    torch.onnx.export(
        model,
        (
            image_embeddings,
            high_res_features1,
            high_res_features2,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
            orig_im_size
        ),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
    )
    original_model = onnx.load(onnx_path)
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def preprocess_image(image, shape):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (input_img / 255.0 - mean) / std
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor


def import_onnx(args):
    onnx_path = args.outdir
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

    encoder_path = os.path.join(onnx_path, "image_encoder.onnx")
    encoder_session = onnxruntime.InferenceSession(encoder_path, providers=providers)

    decoder_path = os.path.join(onnx_path, "image_decoder.onnx")
    decoder_session = onnxruntime.InferenceSession(decoder_path, providers=providers)

    encoder_input_names = [ele.name for ele in encoder_session.get_inputs()]
    encoder_output_names = [ele.name for ele in encoder_session.get_outputs()]
    input_size = encoder_session.get_inputs()[0].shape[2:]

    decoder_input_names = [ele.name for ele in decoder_session.get_inputs()]
    decoder_output_names = [ele.name for ele in decoder_session.get_outputs()]

    image = cv2.imread(args.image)
    image_size = image.shape[:2]
    input_tensor = preprocess_image(image, (input_size[1], input_size[0]))

    image_embeddings, high_res_features1, high_res_features2 = encoder_session.run(
        encoder_output_names, {encoder_input_names[0]: input_tensor}
    )

    mask_input = np.zeros(
        (
            1,
            1,
            input_size[0] // 4,
            input_size[1] // 4,
        ),
        dtype=np.float32,
    )
    has_mask_input = np.array([0], dtype=np.float32)

    points = []
    labels = []
    points.extend([[1215, 125], [1723, 561]])
    points.extend([[0, 0]] * 6)
    labels.extend([2, 3])
    labels.extend([-1] * 6)
    decode(
        os.path.join(args.outdir, "mask_box.png"),
        mask_input,
        has_mask_input,
        input_size,
        image_size,
        decoder_session,
        decoder_input_names,
        decoder_output_names,
        points,
        labels,
        image_embeddings, 
        high_res_features1,
        high_res_features2
    )

    points = []
    labels = []
    points.extend([[1255, 360], [1500, 420], [1654, 459]])
    points.extend([[-1, -1]] * 5)
    labels.extend([1, 1, 1])
    labels.extend([-1] * 5)
    decode(
        os.path.join(args.outdir, "mask_points.png"),
        mask_input,
        has_mask_input,
        input_size,
        image_size,
        decoder_session,
        decoder_input_names,
        decoder_output_names,
        points,
        labels,
        image_embeddings,
        high_res_features1,
        high_res_features2
    )


def decode(mask_path, mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2):
    points, labels = np.array(points), np.array(labels)
    input_point_coords, input_point_labels = prepare_points(
        points, labels, image_size, input_size
    )
    orig_im_size = np.array(image_size, dtype=np.int64)
    inputs = [
        image_embeddings, 
        high_res_features1, 
        high_res_features2,
        input_point_coords, 
        input_point_labels, 
        mask_input, 
        has_mask_input, 
        orig_im_size
    ]
    print(input_point_coords.shape)
    print(input_point_labels.shape)
    print(orig_im_size.shape)
    print(mask_input.shape)
    print("decoder start")
    start = time.perf_counter()
    outputs = sessionDecoder.run(
        output_names_decoder,
        {
            input_names_decoder[i]: inputs[i]
            for i in range(len(input_names_decoder))
        },
    )
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
    masks = outputs[0]
    scores = outputs[1]
    low_res_masks = outputs[2]
    max_idx = np.argmax(scores[0])
    mask = masks[0][max_idx]
    mask[mask > 0.0] = 255
    cv2.imwrite(mask_path, mask)
    low_res_mask = low_res_masks[0][max_idx]
    low_res_mask = np.array([[low_res_mask]])
    return low_res_mask


def prepare_points(
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
        image_size, input_size
) -> tuple[np.ndarray, np.ndarray]:
    input_point_coords = point_coords[np.newaxis, ...]
    input_point_labels = point_labels[np.newaxis, ...]

    input_point_coords[..., 0] = (
        input_point_coords[..., 0]
        / image_size[1]
        * input_size[1]
    )
    
    input_point_coords[..., 1] = (
        input_point_coords[..., 1]
        / image_size[0]
        * input_size[0]
    )

    return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)


def export_onnx(args):
    print(args.config)
    print(args.checkpoint)
    sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")

    image_encoder = ImageEncoder(sam2_model)
    export_image_encoder(image_encoder, args.outdir)

    image_decoder = ImageDecoder(sam2_model)
    export_image_decoder(image_decoder, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export sam2 onnx")
    parser.add_argument("--outdir", type=str, help="path")
    parser.add_argument("--config", type=str, help="*.yaml")
    parser.add_argument("--checkpoint", type=str, required=False,help="*.pt")
    parser.add_argument("--mode", type=str, default="export", required=False, help="export or import")
    parser.add_argument("--image", type=str, required=False, help="image path")
    args = parser.parse_args()
    if args.mode == "export":
        export_onnx(args)
    else:
        import_onnx(args)

