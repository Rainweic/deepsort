import mxnet
import tracking
import cv2 as cv
import numpy as np
from mxnet import nd
from gluoncv import model_zoo, utils, data

# load net
net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mxnet.gpu())

# read video
video_path = "/home/rainweic/桌面/test.mp4"

cap = cv.VideoCapture(video_path)
width, height = 910, 512    # 720P resize to 512x910

thresh = 0.7
tracker = tracking.deepsort(width, height)

while (cap.isOpened()):
    ret, frame = cap.read()

    # 预处理图片 将720p缩放为512x910
    x, img = data.transforms.presets.ssd.transform_test(nd.array(frame),
                                                        short=512)
    class_IDs, scores, bounding_boxes = net(x.as_in_context(mxnet.gpu()))

    scores = scores.asnumpy()[0]
    indexs = np.where(scores >= thresh)[0]
    bounding_boxes = bounding_boxes.asnumpy()[0][indexs]
    # 防止检测到的Box超出图片范围
    if (len(bounding_boxes) > 0):
        bounding_boxes[:, 0:2] = np.maximum(bounding_boxes[:, 0:2], 0)
        bounding_boxes[:, 2] = np.minimum(bounding_boxes[:, 2], width)
        bounding_boxes[:, 3] = np.minimum(bounding_boxes[:, 3], height)

    out = tracker.update(bounding_boxes, img)

    if (len(out) > 0):
        for box in out:
            img = cv.rectangle(
                img,
                (int(box[1]), int(box[2])),
                (int(box[3]), int(box[4])),
                (255,0,0))

    cv.imshow("test", img)
    cv.waitKey(1)


