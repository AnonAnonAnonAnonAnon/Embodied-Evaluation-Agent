from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

class ModelScope:
    def __init__(self):
        # 直接用仓库名，首次会自动从云端下载并缓存到本地
        # 模型页：https://modelscope.cn/models/iic/text-to-video-synthesis
        self.p = pipeline(
            task='text-to-video-synthesis',
            model='iic/text-to-video-synthesis'
        )

    def predict(self, prompt, save_name):
        outputs = self.p({'text': prompt}, output_video=save_name)
        return outputs[OutputKeys.OUTPUT_VIDEO]
