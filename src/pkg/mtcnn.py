import os
import numpy as np
import torch
import cv2

from PIL import Image

from torch import nn
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms

from .utils import download_url_to_file

######################################################################

class PNet(nn.Module):
    """ MTCNN PNet
    
    Keyword Arguments
    - pretrained {bool}: Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            load_weights(self, 'pnet')
            ## state_dict_path = os.path.join(os.path.dirname(__file__),
            ##                                '..', 'data', 'checkpoints', 'pnet.pt')
            ## state_dict = torch.load(state_dict_path)
            ## self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """ MTCNN RNet
    
    Keyword Arguments
    - pretrained {bool}: Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            load_weights(self, 'rnet')
            # state_dict_path = os.path.join(os.path.dirname(__file__),
            #                                '..', 'data', 'checkpoints', 'rnet.pt')
            # state_dict = torch.load(state_dict_path)
            # self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """ MTCNN ONet
    
    Keyword Arguments
    - pretrained {bool}: Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            load_weights(self, 'onet')
            # state_dict_path = os.path.join(os.path.dirname(__file__),
            #                                '..', 'data', 'checkpoints', 'onet.pt')
            # state_dict = torch.load(state_dict_path)
            # self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    """ MTCNN face detection module

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include
    the face only, given raw input images of one of the following types:
    - PIL image or list of PIL images
    - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file also.
    
    Keyword Arguments
    - image_size {int}: Output image size in pixels. The image will be square. (default: {160})
    - margin {int}: Margin to add to bounding box, in terms of pixels in the final image. 
        Note that the application of the margin differs slightly from the davidsandberg/facenet
        repo, which applies the margin to the original image before resizing, making the margin
        dependent on the original image size (this is a bug in davidsandberg/facenet).
        (default: {0})
    - min_face_size {int}: Minimum face size to search for. (default: {20})
    - thresholds {list}: MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
    - factor {float}: Factor used to create a scaling pyramid of face sizes. (default: {0.709})
    - post_process {bool}: Whether or not to post process images tensors before returning.
        (default: {True})
    - select_largest {bool}: If True, if multiple faces are detected, the largest is returned.
        If False, the face with the highest detection probability is returned.
        (default: {True})
    - selection_method {string}: Which heuristic to use for selection. Default None.
        If specified, will override select_largest:
            "probability": highest probability selected
            "largest": largest box selected
            "largest_over_theshold": largest box over a certain probability selected
            "center_weighted_size": box size minus weighted squared offset from image center
        (default: {None})
    - keep_all {bool}: If True, all detected faces are returned, in the order dictated by the
        select_largest parameter. If a save_path is specified, the first face is saved to that
        path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        (default: {False})
    - device {torch.device} -- The device on which to run neural net passes. Image tensors and
        models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(self, image_size=160, margin=0, min_face_size=32,
                 thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                 select_largest=True, selection_method=None, keep_all=False, device=None):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = 'largest' if self.select_largest else 'probability'

    def forward(self, img, save_path=None, return_prob=False):
        """ Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.
        
        Arguments
        - img {PIL.Image, np.ndarray, or list}: A PIL image, np.ndarray, torch.Tensor, or list.
        
        Keyword Arguments
        - save_path {str}: An optional save path for the cropped image. Note that when
            self.post_process=True, although the returned tensor is post processed, the saved
            face image is not, so it is a true representation of the face in the input image.
            If `img` is a list of images, `save_path` should be a list of equal length.
            (default: {None})
        - return_prob {bool}: Whether or not to return the detection probability.
            (default: {False})
        
        Returns
        - Union[torch.Tensor, tuple(torch.tensor, float)]: If detected, cropped image of a face
            with dimensions 3 x image_size x image_size. Optionally, the probability that a
            face was detected. If self.keep_all is True, n detected faces are returned in an
            n x 3 x image_size x image_size tensor with an optional list of detection
            probabilities. If `img` is a list of images, the item(s) returned have an extra 
            dimension (batch) as the first dimension.
        """

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
                )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        if return_prob:
            return faces, batch_probs
        else:
            return faces

    def detect(self, img, landmarks=False):
        """ Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.
        
        Arguments
        - img {PIL.Image, np.ndarray, or list}: A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments
        - landmarks {bool}: Whether to return facial landmarks in addition to bounding boxes.
            (default: {False})
        
        Returns:
        - tuple(numpy.ndarray, list): For N detected faces, a tuple containing an
            Nx4 array of bounding boxes and a length N list of detection probabilities.
            Returned boxes will be sorted in descending order by detection probability if
            self.select_largest=False, otherwise the largest face will be returned first.
            If `img` is a list of images, the items returned have an extra dimension
            (batch) as the first dimension. Optionally, a third item, the facial landmarks,
            are returned if `landmarks=True`.
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
                )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes, dtype='object')
        probs = np.array(probs, dtype='object')
        points = np.array(points, dtype='object')

        if (not isinstance(img, (list, tuple)) and 
            not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
            not (isinstance(img, torch.Tensor) and len(img.shape) == 4)):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if landmarks:
            return boxes, probs, points

        return boxes, probs

    def select_boxes(self, all_boxes, all_probs, all_points, imgs,
                     method='probability', threshold=0.9, center_weight=2.0):
        """ Selects a single box from multiple for a given image using one of multiple heuristics.

        Arguments
        - all_boxes {np.ndarray}: Ix0 ndarray where each element is a Nx4 ndarry of
            bounding boxes for N detected faces in I images (output from self.detect).
        - all_probs {np.ndarray}: Ix0 ndarray where each element is a Nx0 ndarry of
            probabilities for N detected faces in I images (output from self.detect).
        - all_points {np.ndarray}: Ix0 ndarray where each element is a Nx5x2 array of
            points for N detected faces. (output from self.detect).
        - imgs {PIL.Image, np.ndarray, or list}: A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments
        - method {str}: Which heuristic to use for selection:
            "probability": highest probability selected
            "largest": largest box selected
            "largest_over_theshold": largest box over a certain probability selected
            "center_weighted_size": box size minus weighted squared offset from image center
            (default: {'probability'})
        - threshold {float}: Theshold for "largest_over_threshold" method. (default: {0.9})
        - center_weight {float}: Weight for squared offset in center weighted size method.
            (default: {2.0})

        Returns
        - tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): nx4 ndarray of bounding boxes
            for n images. Ix0 array of probabilities for each box, array of landmark points.
        """

        # copying batch detection from extract,
        # but would be easier to ensure detect creates consistent output.
        batch_mode = True
        if (not isinstance(imgs, (list, tuple)) and
            not (isinstance(imgs, np.ndarray) and len(imgs.shape) == 4) and
            not (isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4)):
            imgs = [imgs]
            all_boxes = [all_boxes]
            all_probs = [all_probs]
            all_points = [all_points]
            batch_mode = False

        selected_boxes, selected_probs, selected_points = [], [], []
        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):

            boxes = np.array(boxes, dtype='object')
            probs = np.array(probs, dtype='object')
            points = np.array(points, dtype='object')

            if len(boxes) == 0:
                selected_boxes.append(None)
                selected_probs.append([None])
                selected_points.append(None)
                continue
            elif method == 'largest':
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] \
                        - boxes[:, 1]))[::-1]
            elif method == 'probability':
                box_order = np.argsort(probs)[::-1]
            elif method == 'center_weighted_size':
                box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                img_center = (img.width / 2, img.height/2)
                box_centers = np.array(list(
                    zip((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)
                    ))
                offsets = box_centers - img_center
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
                box_order = np.argsort(box_sizes - offset_dist_squared * center_weight)[::-1]
            elif method == 'largest_over_threshold':
                box_mask = probs > threshold
                boxes = boxes[box_mask]
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] \
                        - boxes[:, 1]))[::-1]
                if sum(box_mask) == 0:
                    selected_boxes.append(None)
                    selected_probs.append([None])
                    selected_points.append(None)
                    continue

            box = boxes[box_order][[0]]
            prob = probs[box_order][[0]]
            point = points[box_order][[0]]
            selected_boxes.append(box)
            selected_probs.append(prob)
            selected_points.append(point)

        if batch_mode:
            selected_boxes = np.array(selected_boxes, dtype='object')
            selected_probs = np.array(selected_probs, dtype='object')
            selected_points = np.array(selected_points, dtype='object')
        else:
            selected_boxes = selected_boxes[0]
            selected_probs = selected_probs[0][0]
            selected_points = selected_points[0]

        return selected_boxes, selected_probs, selected_points

    def extract(self, img, batch_boxes, save_path):
        # Determine if a batch or single image was passed
        batch_mode = True
        if (not isinstance(img, (list, tuple)) and
            not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
            not (isinstance(img, torch.Tensor) and len(img.shape) == 4)):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        # Process all bounding boxes
        faces = []
        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path,
                                    self.post_process)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]

            faces.append(faces_im)

        if not batch_mode:
            faces = faces[0]

        return faces


######################################################################

def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        if isinstance(imgs, np.ndarray):
            imgs = torch.as_tensor(imgs.copy(), device=device)

        if isinstance(imgs, torch.Tensor):
            imgs = torch.as_tensor(imgs, device=device)

        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        imgs = np.stack([np.uint8(img) for img in imgs])
        imgs = torch.as_tensor(imgs.copy(), device=device)
    
    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []

    scale_picks = []

    all_i = 0
    offset = 0
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)
    
        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)

    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    y, ey, x, ex = pad(boxes, w, h)
    
    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, rnet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        
        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, onet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    image_inds = image_inds.cpu()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes = np.array(batch_boxes, dtype='object')
    batch_points = np.array(batch_points, dtype='object')

    return batch_boxes, batch_points


def extract_face(img, box, image_size=160, margin=0, save_path=None, post_process=True):
    """ Extract face + margin from PIL Image given bounding box.
    
    Arguments
    - img {PIL.Image}: A PIL Image.
    - box {numpy.ndarray}: Four-element bounding box.
    - image_size {int}: Output image size in pixels. The image will be square.
    - margin {int}: Margin to add to bounding box, in terms of pixels in the final image. 
        Note that the application of the margin differs slightly from the davidsandberg/facenet
        repo, which applies the margin to the original image before resizing, making the margin
        dependent on the original image size.
    - save_path {str}: Save path for extracted face image. (default: {None})
    - post_process {bool}: Whether or not to post process images tensors before returning.
        Note that when post_process=True, although the returned tensor is post processed,
        the saved face image is not. (default: {True})
    
    Returns
    - torch.tensor: tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
        ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))

    if post_process:
        face = fixed_image_standardization(face)

    return face


######################################################################

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y


def fixed_batch_process(im_data, model):
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i+batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))


def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = torch.nonzero(mask)
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == "Min":
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    # Strategy: in order to perform NMS independently per class.
    # We add an offset to all the boxes. The offset is dependent only on the class idx,
    # and is large enough so that boxes from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    
    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
            ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
            ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size

######################################################################

def load_weights(mdl, name):
    """ Download pretrained state_dict and load into model.

    Arguments
    - mdl {torch.nn.Module}: Pytorch model.
    - name {str}: Name of model.
    """
    if name == 'onet':
        path = 'https://github.com/timesler/facenet-pytorch/raw/master/data/onet.pt'
    elif name == 'pnet':
        path = 'https://github.com/timesler/facenet-pytorch/raw/master/data/pnet.pt'
    elif name == 'rnet':
        path = 'https://github.com/timesler/facenet-pytorch/raw/master/data/rnet.pt'

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'checkpoints')
    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        download_url_to_file(path, cached_file)

    state_dict = torch.load(cached_file)
    mdl.load_state_dict(state_dict)

######################################################################