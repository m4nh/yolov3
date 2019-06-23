import argparse
import shutil
import time
from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
from babylon.network import BMessage, SimpleChannel


def detect(
        cfg,
        weights,
        images,
        output='/tmp/yolov3_output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=False,
        channel=None
):
    device = torch_utils.select_device()

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    save_images = False
    dataloader = LoadWebcam(img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]





    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        image_center = np.array([im0.shape[1] * 0.5, im0.shape[0] * 0.5]).astype(float)
        min_score = 0.1
        super_target_radius = 50
        target_labels = [0]
        min_target = None

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # print(detections)

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                # print('%g %ss' % (n, classes[int(c)]), end=', ')

            feasible_targets = []
            min_distance = 1000000


            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                row = [x1, y1, x2, y2, conf, cls_conf, cls]
                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                # plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

                label = int(cls)
                score = conf
                if score < min_score:
                    continue
                tl = np.array([x1, y1])
                br = np.array([x2, y2])
                center = ((tl + br) * 0.5).astype(int)
                color = colors[int(cls)]
                cv2.circle(im0, (center[0], center[1]), 50, color, 10)

                if label in target_labels:
                    feasible_targets.append(row)
                    distance = np.linalg.norm(image_center - center)
                    if distance < min_distance:
                        min_distance = distance
                        min_target = row
                    dd = 20
                    cv2.line(im0, (center[0] - dd, center[1] - dd), (center[0] + dd, center[1] + dd), color, 3)
                    cv2.line(im0, (center[0] + dd, center[1] - dd), (center[0] - dd, center[1] + dd), color, 3)

        cv2.circle(im0, (int(image_center[0]), int(image_center[1])), super_target_radius, (0, 255, 0), 2)

        if min_target is not None:

            tl = np.array(min_target[0:2])
            br = np.array(min_target[2:4])
            center = ((tl + br) * 0.5).astype(int)
            super_target = False

            if min_distance < super_target_radius:
                super_target = True

            color = (255, 255, 255)
            if super_target:
                color = (0, 255, 0)
            cv2.line(im0, (center[0] - dd, center[1] - dd), (center[0] + dd, center[1] + dd), color, 8)
            cv2.line(im0, (center[0] + dd, center[1] - dd), (center[0] - dd, center[1] + dd), color, 8)

            if not super_target:
                message = BMessage(action="target_acquired")
                message.addField("target", np.array(center))
                print("Sending Target:", np.array(center))
                channel.publish(message)
            else:
                message = BMessage(action="super_target_acquired")
                message.addField("target", np.array(center))
                print("Sending Super Target:", np.array(center))
                channel.publish(message)
        else:
            message = BMessage(action="no_target_acquired")
            print("Sending No Target:")
            channel.publish(message)
            pass


        dt = time.time() - t
        # print('Done. (%.3fs)' % dt)


        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    channel = SimpleChannel(topic_name='vision_module')

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            webcam=True,
            channel=channel
        )
