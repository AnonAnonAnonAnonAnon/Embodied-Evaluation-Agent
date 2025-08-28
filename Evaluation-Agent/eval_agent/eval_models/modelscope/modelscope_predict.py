from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class ModelScope:
    def __init__(self):
        self.p = pipeline('text-to-video-synthesis', f'{CUR_DIR}/checkpoints')
    

    def predict(self, prompt, save_name):
        input = {
            'text': prompt,
        }
        self.p(input, output_video=save_name)[OutputKeys.OUTPUT_VIDEO]