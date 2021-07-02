import os
from torch.onnx.symbolic_helper import parse_args
import torch
export_NMS = True
export_to_tensorrt = True

def dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape):
    """Clip boxes dynamically for onnx.

    Since torch.clamp cannot have dynamic `min` and `max`, we scale the
      boxes by 1/max_shape and clamp in the range [0, 1].

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor or torch.Size): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert isinstance(
        max_shape,
        torch.Tensor), '`max_shape` should be tensor of (h,w) for onnx'

    # scale by 1/max_shape
    x1 = x1 / max_shape[1]
    y1 = y1 / max_shape[0]
    x2 = x2 / max_shape[1]
    y2 = y2 / max_shape[0]

    # clamp [0, 1]
    x1 = torch.clamp(x1, 0, 1)
    y1 = torch.clamp(y1, 0, 1)
    x2 = torch.clamp(x2, 0, 1)
    y2 = torch.clamp(y2, 0, 1)

    # scale back
    x1 = x1 * max_shape[1]
    y1 = y1 * max_shape[0]
    x2 = x2 * max_shape[1]
    y2 = y2 * max_shape[0]
    return x1, y1, x2, y2


def get_k_for_topk(k, size):
    """Get k of TopK for onnx exporting.

    The K of TopK in TensorRT should not be a Tensor, while in ONNX Runtime
      it could be a Tensor.Due to dynamic shape feature, we have to decide
      whether to do TopK and what K it should be while exporting to ONNX.
    If returned K is less than zero, it means we do not have to do
      TopK operation.

    Args:
        k (int or Tensor): The set k value for nms from config file.
        size (Tensor or torch.Size): The number of elements of \
            TopK's input tensor
    Returns:
        tuple: (int or Tensor): The final K for TopK.
    """
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if torch.onnx.is_in_onnx_export():
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if is_trt_backend:
            # TensorRT does not support dynamic K with TopK op
            if 0 < k < size:
                ret_k = k
        else:
            # Always keep topk op for dynamic input in onnx for ONNX Runtime
            ret_k = torch.where(k < size, k, size)
    elif k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k


def add_dummy_nms_for_onnx(boxes,
                           scores,
                           max_output_boxes_per_class=1000,
                           iou_threshold=0.5,
                           score_threshold=0.05,
                           pre_top_k=-1,
                           after_top_k=-1,
                           labels=None):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes]
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (bool): Number of top K boxes to keep before nms.
            Defaults to -1.
        after_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        labels (Tensor, optional): It not None, explicit labels would be used.
            Otherwise, labels would be automatically generated using
            num_classed. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: dets of shape [N, num_det, 5] and class labels
            of shape [N, num_det].
    """
    pre_top_k = 1000

    if export_to_tensorrt:
        max_output_boxes_per_class = torch.tensor(max_output_boxes_per_class, dtype=torch.int32)
    else:
        max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    num_class = scores.shape[2]

    nms_pre = torch.tensor(pre_top_k, device=scores.device, dtype=torch.long)
    nms_pre = get_k_for_topk(nms_pre, boxes.shape[1])

    if nms_pre > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(nms_pre)
        batch_inds = torch.arange(batch_size).view(
            -1, 1).expand_as(topk_inds).long()
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        transformed_inds = boxes.shape[1] * batch_inds + topk_inds
        boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
            batch_size, -1, 4)
        scores = scores.reshape(-1, num_class)[transformed_inds, :].reshape(
            batch_size, -1, num_class)
        if labels is not None:
            labels = labels.reshape(-1, 1)[transformed_inds].reshape(
                batch_size, -1)
    if not export_NMS:
        # dets = torch.cat([boxes[:, 0:after_top_k, :], scores[:, 0:after_top_k, 1].unsqueeze(-1)], dim=2)
        # labels = torch.ones(boxes.shape[0], after_top_k, dtype=torch.long).to(scores.device)
        # return dets, labels
        print("export_NMS=False")
        boxes = boxes.unsqueeze(2)
        # scores = scores.transpose(1, 2)
        return boxes, scores

    scores = scores.permute(0, 2, 1)
    num_box = boxes.shape[1]
    # turn off tracing to create a dummy output of nms
    state = torch._C._get_tracing_state()

    torch.manual_seed(0)
    if export_to_tensorrt:
        num_fake_det = 2
        dummy_num_detections = torch.tensor([[num_fake_det]]).expand(batch_size, 1)  # [1,1]
        dummy_boxes = torch.rand(batch_size, num_fake_det, 4)
        dummy_scores = torch.rand(batch_size, num_fake_det)
        # dummy_labels = torch.randint(num_class, (batch_size, num_fake_det)) 
        # dummy_labels = dummy_labels.type(torch.float32)
        dummy_labels = torch.tensor([[74, 19]], dtype=torch.float32)
        setattr(DummyONNXNMSop, 'output', (dummy_num_detections, dummy_boxes, dummy_scores, dummy_labels))
    else:
        # dummy indices of nms's output
        num_fake_det = 2
        batch_inds = torch.randint(batch_size, (num_fake_det, 1))
        cls_inds = torch.randint(num_class, (num_fake_det, 1))
        box_inds = torch.randint(num_box, (num_fake_det, 1))
        indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
        output = indices
        setattr(DummyONNXNMSop, 'output', output)
        # setattr(DummyONNXNMSop, 'output', (output, torch.Tensor([3,4])))
    


    # num_fake_det = 2
    # dummy_num_detections = torch.tensor([[num_fake_det]]).expand(batch_size, 1)  # [1,1]
    # # scores_ind = torch.randint(scores.numel(), (num_fake_det,))
    # scores_ind = torch.tensor([7304, 7369], dtype=torch.int32) # # np.argwhere((scores.flatten()> 0.2))
    # mask = torch.zeros(scores.numel(), dtype=torch.int32)
    # for ind in scores_ind:
    #     mask[ind] += 1
    # mask = mask.view(scores.shape)
    # dummy_scores = torch.masked_select(scores, mask>0)
    # dummy_scores = dummy_scores.unsqueeze(0) # TODO
    # boxes_mask = mask.unsqueeze()
    # boxes_mask = boxes_mask.expand(*boxes_mask.shape, 4)
    # dummy_boxes = torch.masked_selected(boxes, boxes_mask)
    # dummy_boxes = dummy_boxes.unsqueeze(0)
    # labels = torch.range(scores.numel).view(scores.shape)%scores.shape[1]//scores.shape[2]
    # dummy_labels = torch.masked_select(labels, mask)
    # dummy_labels = torch.unsqueeze(0)
    # setattr(DummyONNXNMSop, 'output', (dummy_num_detections, d ummy_boxes, dummy_scores, dummy_labels))


    # open tracing
    torch._C._set_tracing_state(state)

    if not export_NMS:
        pass
    if export_to_tensorrt:
        # convert input to BatchNMS_TRT
        boxes = boxes.unsqueeze(2)
        scores = scores.transpose(1, 2)
        num_detections, boxes, scores, labels = DummyONNXNMSop.apply(boxes, scores, # boxes:([1, 3652, 4]
                                                max_output_boxes_per_class,
                                                iou_threshold, score_threshold)

        scores = scores.unsqueeze(2)          # [1, 200, 1]
        dets = torch.cat([boxes, scores], dim=2)   #  [1, 200, 5]

    else:
        selected_indices = DummyONNXNMSop.apply(boxes, scores, # boxes:([1, 3652, 4]
                                                max_output_boxes_per_class,
                                                iou_threshold, score_threshold)
        # convert output from BatchNMS_TRT
        batch_inds, cls_inds = selected_indices[:, 0], selected_indices[:, 1]
        box_inds = selected_indices[:, 2]
        if labels is None:
            labels = torch.arange(num_class, dtype=torch.long).to(scores.device)
            labels = labels.view(1, num_class, 1).expand_as(scores) # ([1, 80, 3652])
        scores = scores.reshape(-1, 1)                              # [1*80*3652, 1]
        boxes = boxes.reshape(batch_size, -1).repeat(1, num_class).reshape(-1, 4) # [1*80*3652, 4]
        pos_inds = (num_class * batch_inds + cls_inds) * num_box + box_inds  # box index in all data [2]
        mask = scores.new_zeros(scores.shape)    # [80*3652, 1]
        # mask = torch.zeros(scores.shape[0], scores.shape[1], dtype=torch.float32)
        # mask = torch.zeros_like(scores)
        # mask = torch.rand(scores.shape, dtype=torch.float32) # fail
        # mask[:, :] = 0
        # mask = torch.rand(scores.shape, dtype=torch.float32)
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        # PyTorch style code: mask[batch_inds, box_inds] += 1
        #mask[pos_inds, :] += 1
        for i in pos_inds:
            mask[i, 0] += 1
        scores = scores * mask
        boxes = boxes * mask
   
        scores = scores.reshape(batch_size, -1) # [1, 80*3652]
        boxes = boxes.reshape(batch_size, -1, 4) # [1, 80*3652, 4]
        labels = labels.reshape(batch_size, -1) # [1, 80*3652]

        nms_after = torch.tensor(
            after_top_k, device=scores.device, dtype=torch.long)
        nms_after = get_k_for_topk(nms_after, num_box * num_class)

        if nms_after > 0:
            _, topk_inds = scores.topk(nms_after)   # [1, 200]
            batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds) # [1, 200]
            # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
            transformed_inds = scores.shape[1] * batch_inds + topk_inds   # [1, 200]
            scores = scores.reshape(-1, 1)[transformed_inds, :].reshape(
                batch_size, -1)            # [1, 200]
            boxes = boxes.reshape(-1, 4)[transformed_inds, :].reshape(
                batch_size, -1, 4)         #  [4, 200, 4]
            labels = labels.reshape(-1, 1)[transformed_inds, :].reshape(
                batch_size, -1)            # [1, 200]

        scores = scores.unsqueeze(2)          # [1, 200, 1]
        dets = torch.cat([boxes, scores], dim=2)   #  [1, 200, 5]

    
        # dets = selected_indices[:, 2].unsqueeze(0)
        # labels = selected_indices[:, 1].unsqueeze(0)
    return dets, labels


class DummyONNXNMSop(torch.autograd.Function):
    """DummyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):

        return DummyONNXNMSop.output

    @staticmethod
    @parse_args('v', 'v', 'v', 'v', 'v')
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        if export_to_tensorrt:
            return g.op(
                'mydomain::BatchedNMS_TRT', 
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                outputs=4)
        else:
            return g.op(
                'NonMaxSuppression', 
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                outputs=1)



    