# Hyperparanms
HEIGHT = 128
WIDTH = 128
CHANNELS = 4
FPS = 10
DURATION = 3 
SEQ_LEN = FPS * DURATION

# MVO Prediction Logic Mapping
# FRONT: 0, LEFT: 1, RIGHT: 2
def get_label_id(label_name):
    mapping = {'front': 0, 'left': 1, 'right': 2}
    return mapping.get(label_name.lower(), 0)

# Paths based on your setup
VIDEO_DIR = r''
LABEL_DIR = r''

# Intent files
VAL_POSITIONS = ''
TEST_POSITIONS = ''