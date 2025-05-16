import boto3
import cv2
import os
import credentials

# folders
output_dir = 'data'
output_dir_imgs = os.path.join(output_dir, 'imgs')
output_dir_boxes = os.path.join(output_dir, 'boxes')
output_dir_anns = os.path.join(output_dir, 'anns')

# in case they don't exist
os.makedirs(output_dir_imgs, exist_ok=True)
os.makedirs(output_dir_anns, exist_ok=True)
os.makedirs(output_dir_boxes, exist_ok=True)


# create aws rekognition client
reko_client = boto3.client('rekognition',
                           aws_access_key_id = credentials.access_key,
                            aws_secret_access_key= credentials.secret_key)

# target class
target_class = 'Person'

# load video
video = cv2.VideoCapture('subway.mp4')
frame_nmr = -1

# read frames
flag = True
while flag:
    flag, frame = video.read()

    if flag:

        frame_nmr += 1

        # process 1 out of 3 frames
        if frame_nmr % 3 != 0:
            continue

        # save raw frames
        raw_frame = frame.copy()

        # preparing for aws rekognition
        H, W, _ = frame.shape
        # convert frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)
        # convert buffer to bytes
        frame_bytes = buffer.tobytes()

        # detect objects
        response = reko_client.detect_labels(Image={'Bytes' : frame_bytes},
                                            MinConfidence = 50)

        ann_path = os.path.join(output_dir_anns, f'frame_{frame_nmr:06d}.txt')

        with open(ann_path, 'w') as f:

            for label in response['Labels']:

                if label['Name'] == target_class:

                    for instance_nmr in range(len(label['Instances'])):
                        bound_box = label['Instances'][instance_nmr]['BoundingBox']
                        x = bound_box['Left']
                        y = bound_box['Top']
                        w = bound_box['Width']
                        h = bound_box['Height']

                        # write detections (anns)
                        f.write('{} {} {} {} {}\n'.format(0,
                                                          (x + w / 2),
                                                          (y + h / 2),
                                                          w,
                                                          h))

                        # convert to pixel coordinates
                        x1 = int(x * W)
                        y1 = int(y * H)
                        x2 = int((x + w) * W)
                        y2 = int((y + h) * H)

                        # bounding boxes
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # save original frames
        raw_frame_path = os.path.join(output_dir_imgs, f'frame_{frame_nmr:06d}.jpg')
        cv2.imwrite(raw_frame_path, raw_frame)

        # save frames with boxes
        processed_frame_path = os.path.join(output_dir_boxes, f'frame_{frame_nmr:06d}.jpg')
        cv2.imwrite(processed_frame_path, frame)