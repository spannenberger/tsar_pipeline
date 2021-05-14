from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
from google.protobuf import text_format
from pprint import pprint
import yaml

@Registry
class TritonConfigCreator(Callback):

    def __init__(self, conf_path=''):
        super().__init__(CallbackOrder.Internal)
        self.conf_path = Path(conf_path)
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        with open('config/classification/augmentations/light.yml', encoding="utf8") as f:
            params = yaml.safe_load(f)
            height = params['train']['transforms'][1]['height']
            width = params['train']['transforms'][1]['width']
        
        with open(self.conf_path, "a") as fw:
            fw.write('platform: "onnxruntime_onnx"\n')
            fw.write('input [\n')
            fw.write('{\n')
            fw.write(f'\tname: "input"\n\tdata_type: TYPE_FP32\n\tdims: [-1, 3, {height}, {width}]\n')
            fw.write('}\n')
            fw.write(']\n')
            fw.write('output [\n')
            fw.write('{\n')
            fw.write('\tname: "output"\n\tdata_type: TYPE_FP32\n\tdims: [-1, 18]\n}\n')
            fw.write(']\n')
            fw.write('instance_group [\n')
            fw.write('{\n')
            fw.write('\tcount: 1\n\tkind: KIND_GPU\n\tgpus: [ 0 ]\n}\n')
            fw.write(']\n')