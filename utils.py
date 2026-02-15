# Hyperparanms
HEIGHT = 128
WIDTH = 128
CHANNELS = 6  # Proto 1: 6-channel (RGB + Intent)
# CHANNELS = 3  # Proto 2: 3-channel RGB-only (Eric's simplified version)
FPS = 10
DURATION = 3 
SEQ_LEN = FPS * DURATION

# MVO Prediction Logic Mapping
# FRONT: 0, LEFT: 1, RIGHT: 2
def get_label_id(label_name):
    mapping = {'front': 0, 'left': 1, 'right': 2}
    return mapping.get(label_name.lower(), 0)

# Paths based on your setup
# Proto 1: Splitfolders approach (ACTIVE)
# Contains both and only video and label directories
# Folder names are strictly "videos" and "labels"
DATA_DIR = r'D:\Thesis 2\Thesis 2\AIGD\split folder' 

# Intent files
VAL_POSITIONS = ''
TEST_POSITIONS = ''

# Proto 2: In-memory splitting approach (Eric's version - COMMENTED OUT)
# VIDEO_DIR = r''
# LABEL_DIR = r''