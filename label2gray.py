import glob
import os.path
import numpy as np
from PIL import Image
import tensorflow as tf

# FLAGS = tf.compat.v1.flags.FLAGS
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('original_gt_folder',  # 读取voc格式的png图片路径
                       'data/voc_train_segmentation',  # default
                       'Original ground truth annotations.')  # help
tf.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')
tf.flags.DEFINE_string('output_dir',  # 保存路径
                       'data/train_segmentation1',
                       'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
    """Removes the color map from the annotation.
    Args:
      filename: Ground truth annotation filename.
    Returns:
      Annotation without color map.
    """
    return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
    """Saves the annotation as png file.
    Args:
      annotation: Segmentation annotation.
      filename: Output filename.
    """

    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    '''
    with tf.io.gfile.GFile(filename, mode='w') as f:
    #with open(filename, mode='w') as f:
      print(f)
      pil_image.save(f, 'PNG')
      '''
    pil_image.save(filename)


def main(unused_argv):
    # Create the output directory if not exists.
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    # if not tf.io.gfile.isdir(FLAGS.output_dir):
    # tf.io.gfile.makedirs(FLAGS.output_dir)

    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                         '*.' + FLAGS.segmentation_format))
    for annotation in annotations:
        raw_annotation = _remove_colormap(annotation)
        filename = os.path.basename(annotation)[:-4]
        _save_annotation(raw_annotation,
                         os.path.join(
                             FLAGS.output_dir,
                             filename + '.' + FLAGS.segmentation_format))


if __name__ == '__main__':
    # tf.compat.v1.app.run()
    tf.app.run()
