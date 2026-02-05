import os

def run_tensorboard(logdir='lightning_logs', port=6006):
    """
    Starts TensorBoard so you can view training graphs.
    """
    # Simply runs the command line instruction to start tensorboard
    os.system(f'tensorboard --logdir={logdir} --port={port}')

if __name__ == '__main__':
    run_tensorboard()