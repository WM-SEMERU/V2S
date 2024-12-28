import subprocess as sp
import os
import json
import argparse
import numpy as np
import os
import glob
import sys
import tensorflow as tf
import datetime
import json
import ntpath
from PIL import ImageFile
from PIL import Image

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--object_path",
                    help="Path to object detection library")  # which also happens to be working directory
parser.add_argument("--model_path", help="Path to frozen graph")  # trained ML model
args = parser.parse_args()

sys.path.append(args.object_path)

# import the local tensorflow object detection library;
# cannot be done unless object_path in sys.path
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


class ScreenDetection:
    def __init__(self, screenId):
        self.screenId = screenId
        self.screenTap = []

    def addTap(self, screenTap):
        self.screenTap.append(screenTap)

    def asJson(self):
        return dict(screenId=self.screenId, screenTap=self.screenTap)


class ScreenTap:
    def __init__(self, x, y, confidence):
        self.x = x
        self.y = y
        self.confidence = confidence

    def asJson(self):
        return dict(x=self.x, y=self.y, confidence=self.confidence)


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'asJson'):
            return obj.asJson()
        else:
            return json.JSONEncoder.default(self, obj)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(args.object_path, args.model_path)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(args.object_path, "data/model/v2s_label_map.pbtxt")
NUM_CLASSES = 1


def fix_video_frame_rate(video_path=None):
    """ Fixes videos in video_path.
    Creates a folder for each video to contain other intermediate info.
    video encoding: -vcodec libx264
    override output: -y
    -preset ultrafast
    constant rate factor: -crf 15 """

    if not video_path:
        video_path = os.path.join(args.object_path, "data/videos/")

    # get all files
    files = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]

    for f in files:
        video_name, video_extension = os.path.splitext(f)

        if video_extension == ".mp4":  # make sure we're only dealing with mp4 video files
            flder_name = os.path.join(video_path, video_name)
            if not os.path.exists(flder_name):
                os.mkdir(flder_name)  # create output folder for each video
                print(flder_name)

            command = 'ffmpeg -y -i ' + flder_name + '.mp4 -vcodec libx264 -r 30 -preset ultrafast -crf 15 ' + \
                      flder_name + '\\' + video_name + '-fixed.mp4 -hide_banner -loglevel panic'
            response = sp.Popen(command, shell=True, stdout=sp.PIPE)
            response.wait()
            response.terminate()


def extract_frames(video_path=None):
    """
    Extract frames from videos with fixed frame rates
    :param video_path: the path to the video containing all the original videos
    """
    if not video_path:
        video_path = os.path.join(args.object_path, "data/videos/")

    # get all sub-directories of video_path
    files = [f for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f))]

    for f in files:
        dir_path = os.path.join(video_path, f)
        fixed_video_path = os.path.normpath(dir_path + "/" + f + "-fixed.mp4")

        # for each folder, create a folder for cropped imaegs
        if not os.path.exists(dir_path + '\\extracted_frames'):
            os.mkdir(dir_path + '\\extracted_frames')

        # split things
        command = 'ffmpeg -i ' + fixed_video_path + ' -qscale:v 3 -vf fps=30 ' + \
                  dir_path + '\\extracted_frames\\%04d.jpg -hide_banner -loglevel panic'

        response = sp.Popen(command, shell=True, stdout=sp.PIPE)
        response.wait()
        response.terminate()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def save_image_array_as_jpg(image, output_path):
    """Saves an image (represented as a numpy array) to JPEG.
        Args:
            image: a numpy array with shape [height, width, 3].
            output_path: path to which image should be written.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.gfile.Open(output_path, 'w') as fid:
        image_pil.save(fid, 'JPEG', quality=80, optimize=True, progressive=True)


def detect_frames(video_path=None):
    """
    Detect touch indicators from extracted frames
    :param video_path:
    :return: output is frames.json
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # label map related 
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    if not video_path:
        video_path = os.path.join(args.object_path, "data/videos/")

    # get all sub-dirs of video_path 
    files = [f for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f))]

    for f in files:
        extracted_frames_dir_path = os.path.join(video_path, f, "extracted_frames")

        extracted_frames = glob.glob(os.path.join(extracted_frames_dir_path, '*'))
        extracted_frames.sort()

        detection_output_path = os.path.join(video_path, f, "detected_frames")

        # Validate detection folder exists
        if not os.path.exists(detection_output_path):
            os.makedirs(detection_output_path)

        print('Analyzing video: [{}]'.format(f))

        json_file = os.path.join(detection_output_path, f, '-frames.json')
        detections = []

        config = tf.ConfigProto(inter_op_parallelism_threads=4,
                                allow_soft_placement=True)

        start_detection_time = datetime.datetime.now().replace(microsecond=0)

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph, config=config) as sess:
                for image_path in progress_bar(extracted_frames, "Computing: ", 40):
                    # print(image_path)
                    # ret, image_np = cap.read()
                    image = Image.open(image_path)
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    # start_time = datetime.datetime.now().replace(microsecond=0)
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # final_time = datetime.datetime.now().replace(microsecond=0)
                    # print('Time: ' + str((final_time - start_time)), flush=True)
                    boxes = boxes[0]
                    scores = scores[0]
                    # Visualization of the results of a detection.
                    # print('========')
                    # print(box)
                    # print(score)
                    base_name = ntpath.basename(image_path)
                    base_name, file_extension = os.path.splitext(base_name)

                    detection = ScreenDetection(int(base_name))
                    (im_width, im_height) = image.size
                    for i in range(len(boxes)):
                        box = boxes[i]
                        score = scores[i]

                        if score > 0.5:
                            # Add detection box on the image
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)

                            yMin = box[0] * im_height
                            xMin = box[1] * im_width
                            yMax = box[2] * im_height
                            xMax = box[3] * im_width
                            x = xMin + ((xMax - xMin) / 2.0)
                            y = yMin + ((yMax - yMin) / 2.0)
                            detection.addTap(ScreenTap(x, y, float(score)))

                    if (len(detection.screenTap) > 0):
                        detections.append(detection)
                    output_image_file = os.path.join(detection_output_path, 'bbox-' + base_name + file_extension)
                    # print('Processing: ' + output_image_file, flush=True)
                    # Don't remove, fixes weird error: https://stackoverflow.com/questions/19600147/sorl-thumbnail-encoder-error-2-when-writing-image-file/41018959#41018959
                    ImageFile.MAXBLOCK = im_width * im_height
                    save_image_array_as_jpg(image_np,
                                            os.path.join(detection_output_path, 'bbox-' + base_name + file_extension))

        end_detection_time = datetime.datetime.now().replace(microsecond=0)
        print('Detections took: ' + str(end_detection_time - start_detection_time) + ' to run.')

        # Save json data
        json_data = json.dumps(detections, cls=ComplexEncoder, sort_keys=True)
        file = open(json_file, "w")
        file.write(json_data)
        file.close()


def progress_bar(data, prefix="", size=50, file=sys.stdout):
    count = len(data)

    def show(curr_prog):
        x = int(size * curr_prog / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), curr_prog, count))
        file.flush()

    show(0)
    for i, item in enumerate(data):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


# Fix videos' frame rates 
print("Fixing video frame rate ... ", end="")
fix_video_frame_rate()
print("Finished!")
print("Extracting frames... ", end="")
extract_frames()
print("Finished!")
detect_frames()
