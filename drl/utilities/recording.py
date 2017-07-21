import os
import shutil
import pyglet
import ffmpy

MAX_LEN = 10
REC_DIR = '/tmp/drl_record'


def record_frame(step):
    if not os.path.exists(REC_DIR):
        os.makedirs(REC_DIR)

    pre_zeros = len(str(MAX_LEN)) - len(str(step))
    path = os.path.join(REC_DIR, '0' * pre_zeros + str(step))
    pyglet.image.get_buffer_manager().get_color_buffer().save(path + '.png')


def save_recording(path):
    # Intialize ffmpy
    ff = ffmpy.FFmpeg(
        inputs={os.path.join(REC_DIR, '*.png'): '-y -framerate 24 -pattern_type glob'},
        outputs={os.path.join(path, 'video.mp4'): None}
    )
    # Send output of ffmpy to log.txt file in temporary record folder
    # if error check the log file
    ff.run(stdout=open(os.path.join(REC_DIR, 'log.txt'), 'w'), stderr=open(os.path.join(REC_DIR, 'tmp.txt'), 'w'))
    # Remove .png and log.txt file
    shutil.rmtree(REC_DIR)