import io
import boto3
import cv2
import uuid
import logging
from botocore.exceptions import ClientError

# create aws clients
reko = boto3.client('rekognition')

s3 = boto3.client('s3')

# create s3 bucket
bucket_name = f'aws-person-rekognition-{uuid.uuid4()}'

# create bucket
try:
    s3.create_bucket(Bucket=bucket_name)
except ClientError as e:
    logging.error(e)
except s3.exceptions.BucketAlreadyExists as b:
    logging.error(b)

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

        # process 1 out of 4 frames
        if frame_nmr % 4 != 0:
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
        response = reko.detect_labels(Image={'Bytes' : frame_bytes},
                                            MinConfidence = 50)

        anns_buffer = io.StringIO()

        for label in response['Labels']:

            if label['Name'] == target_class:

                for instance_nmr in range(len(label['Instances'])):
                    bound_box = label['Instances'][instance_nmr]['BoundingBox']
                    x = bound_box['Left']
                    y = bound_box['Top']
                    w = bound_box['Width']
                    h = bound_box['Height']

                    # write detections (anns)
                    anns_buffer.write(f"0 {(x + w / 2)} {(y + h / 2)} {w} {h}\n")

                    # convert to pixel coordinates
                    x1 = int(x * W)
                    y1 = int(y * H)
                    x2 = int((x + w) * W)
                    y2 = int((y + h) * H)

                    # bounding boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    has_person = True

        # upload original frames
        _, raw_buffer = cv2.imencode('.jpg', raw_frame)
        raw_stream = io.BytesIO(raw_buffer.tobytes())
        s3.upload_fileobj(raw_stream, bucket_name, f"imgs/frame_{frame_nmr:06d}.jpg")

        # upload processed frames
        _, processed_buffer = cv2.imencode('.jpg', frame)
        processed_stream = io.BytesIO(processed_buffer.tobytes())
        s3.upload_fileobj(processed_stream, bucket_name, f"boxes/frame_{frame_nmr:06d}.jpg")

        # upload annotations
        anns_stream = io.BytesIO(anns_buffer.getvalue().encode('utf-8'))
        s3.upload_fileobj(anns_stream, bucket_name, f"anns/frame_{frame_nmr:06d}.txt")