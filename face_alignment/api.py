import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torchvision.transforms as T

import torch
import warnings
from enum import IntEnum
from distutils.version import LooseVersion

from .utils import *

from core_lib.utilities import logger

class LandmarksType(IntEnum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(IntEnum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4


class FaceAlignment:
    def __init__(self, landmarks_type, face_align_model_path, depth_pred_model_path, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', face_detector_kwargs=None, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.face_align_model_path = face_align_model_path
        self.depth_pred_model_path = depth_pred_model_path
        self.verbose = verbose

        if LooseVersion(torch.__version__) < LooseVersion('1.5.0'):
            raise ImportError(f'Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
                            Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0')

        network_size = int(network_size)
        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        face_detector_kwargs = face_detector_kwargs or {}
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)

        # Initialise the face alignemnt networks
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)
        self.face_alignment_net = torch.jit.load(face_align_model_path)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = torch.jit.load(depth_pred_model_path)

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_faces=None, return_bboxes=False, return_landmark_score=False):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces, return_bboxes, return_landmark_score)

    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """
        image = get_image(image_or_path)

        start = time.time()

        ht = image.shape[0]
        wd = image.shape[1]

        # Force detected faces to be OUR bounding boxes
        detected_faces=[np.array([1.0, np.float(ht-1), np.float(wd-1), 1.0])]
        #print(f'detected_faces: {detected_faces}')
        logger.info(f'Landmark detector input image size: {ht} x {wd}')

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image.copy())
        
        final = time.time()
        logger.warning(f'Execution time for detected_faces: {final-start} seconds! ')
        logger.warning(f'{len(detected_faces)} Face(s) Detected!')

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores = []
        for i, d in enumerate(detected_faces):
            logger.info(f'Bounding Box for face detected is {d}')
            
            center = torch.tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12        # PERQUE?? NO ES FA SERVIR MES!
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale
            #scale = 2 #???????

            logger.warning('FACE ALIGNMENT')
            logger.info(f'Input image: {(image).shape}')
            logger.info(f'Extra information: Center {center}, Scale value {scale}  // Reference scale {self.face_detector.reference_scale}')
            
            # image shape -> Size([358, 358, 3])

            # Save image
            im = Image.fromarray(image)
            logger.error("SAVE INPUT IMAGE!!!")
            output_path = os.path.join('/Users/crisalix/Desktop/test/image.jpg')
            im.save(output_path)
            
            '''
            logger.warning('CROP')
            # Crop taking into account center and scale --> Crop + Resize
            inp = crop(image, center, scale)
            logger.info(f'Shape after crop {inp.shape}')
            # inp.shape -> Size([256, 256, 3])

            # Save crop
            logger.error("SAVE CROPPED IMAGE!!!")
            inp_arr = Image.fromarray(inp)
            output_path = os.path.join('/Users/crisalix/Desktop/test/image_2.jpg')
            inp_arr.save(output_path)

            # Transpose and torch!
            inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()
            # inp.shape -> torch.Size([3, 256, 256])
            
            '''

            #Resize + torch + transpose to our
            transform = T.Resize(size = (256,256))
            inp = transform(torch.from_numpy(image.transpose((2, 0, 1))).float())           
            
            inp = inp.to(self.device)       
            inp.div_(255.0).unsqueeze_(0)
            # inp.shape -> torch.Size([1, 3, 256, 256])

            logger.warning("FACE ALIGNMENT NET!!!")
            out = self.face_alignment_net(inp).detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
            out = out.cpu().numpy()

            # Out Face Alignment
            logger.warning("AFTER FACE ALIGNMENT!!!")
            logger.info(f'FA_NET output type: {type(out)} and shape: {out.shape}')
            logger.error(f'SAVE HEATMAP IMAGE!!!')
            #for i in range(len(out[0])):
            fig = plt.figure(figsize=plt.figaspect(.5))
            plt.imshow(out[0][0])
            output_path = os.path.join('/Users/crisalix/Desktop/test/', 'image_hm' + str(0) + '.jpg')
            plt.savefig(output_path, subsampling=0, quality=95)
            plt.close(fig)

            '''
            pts, pts_img, scores = get_preds_fromhm(out, center.numpy(), scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 4 , pts_img.view(68, 2)
            '''

            # New option, not using scale nor center values
            pts, pts_img, scores = get_preds_fromhm(out)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(68, 2) * 5.59375 , pts.view(68, 2) * 5.59375 

            scores = scores.squeeze(0)

            logger.warning("AFTER GET PREDS FROM HM!!!")
            logger.info(f'POINTS PREDS -> pts_img size: {pts_img.shape}')
            
            logger.error(f'SAVE IMAGE AFTER FA!!!')
            fig = plt.figure(figsize=plt.figaspect(.5))

            '''
            transform = T.Compose([
                                    T.ToPILImage(),
                                    T.Resize(size = (256,256))
                                ])
            pts_image = transform(image) '''

            plt.imshow(image) # pts_image
            plt.scatter(pts[:, 0], pts[:, 1], marker="o", color="red", s=2) #pts_img
            output_path = os.path.join('/Users/crisalix/Desktop/test/image_after_FA.jpg')
            plt.savefig(output_path, subsampling=0, quality=95)
            plt.close(fig)

            if self.landmarks_type == LandmarksType._3D:
                logger.warning("LANDMARKS TYPE 3D!!!")
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0 and pts[i, 1] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                logger.info(f'Heatmaps 2 info {heatmaps.shape}')
                
                logger.error(f'SAVE HEATMAP 2 IMAGE!!!')
                fig = plt.figure(figsize=plt.figaspect(.5))
                plt.imshow(heatmaps[0])
                output_path = os.path.join('/Users/crisalix/Desktop/test/', 'image_hm' + str(2) + '.jpg')
                plt.savefig(output_path, subsampling=0, quality=95)
                plt.close(fig)

                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)
                logger.info(f'Heatmap size: {heatmaps.shape}')

                logger.warning("DEPTH PREDICTION!!!")
                heatmaps = heatmaps.to(self.device)
                depth_pred = self.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                logger.info(f'Output depth predictor: {depth_pred.shape}')
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1) #Understand depth_pred modification!!!   

            logger.info(f'Last output: {pts_img.shape}')
            logger.info(f'Last output: {pts_img}')
            landmarks.append(pts_img.numpy())
            landmarks_scores.append(scores)
        
        logger.print('---------------------------------------')

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores, detected_faces
        else:
            return landmarks

    @torch.no_grad()
    def get_landmarks_from_batch(self, image_batch, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_batch {torch.tensor} -- The input images batch

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmarks
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_batch(image_batch)

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores_list = []
        # A batch for each frame
        for i, faces in enumerate(detected_faces):
            res = self.get_landmarks_from_image(
                image_batch[i].cpu().numpy().transpose(1, 2, 0),
                detected_faces=faces,
                return_landmark_score=return_landmark_score,
            )
            if return_landmark_score:
                landmark_set, landmarks_scores, _ = res
                landmarks_scores_list.append(landmarks_scores)
            else:
                landmark_set = res
            # Bacward compatibility
            if landmark_set is not None:
                landmark_set = np.concatenate(landmark_set, axis=0)
            else:
                landmark_set = []
            landmarks.append(landmark_set)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores_list = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores_list, detected_faces
        else:
            return landmarks

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True,
                                     return_bboxes=False, return_landmark_score=False):
        """Scan a directory for images with a given extension type(s) and predict the landmarks for each
            face present in the images found.

         Arguments:
            path {str} -- path to the target directory containing the images

        Keyword Arguments:
            extensions {list of str} -- list containing the image extensions considered (default: ['.jpg', '.png'])
            recursive {boolean} -- If True, scans for images recursively (default: True)
            show_progress_bar {boolean} -- If True displays a progress bar (default: True)
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        """
        detected_faces = self.face_detector.detect_from_directory(path, extensions, recursive, show_progress_bar)

        predictions = {}
        for image_path, bounding_boxes in detected_faces.items():
            image = io.imread(image_path)
            if return_bboxes or return_landmark_score:
                preds, bbox, score = self.get_landmarks_from_image(
                    image, bounding_boxes, return_bboxes=return_bboxes, return_landmark_score=return_landmark_score)
                predictions[image_path] = (preds, bbox, score)
            else:
                preds = self.get_landmarks_from_image(image, bounding_boxes)
                predictions[image_path] = preds

        return predictions
