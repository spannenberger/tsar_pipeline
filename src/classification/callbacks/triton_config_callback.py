from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
from google.protobuf import text_format
from pprint import pprint
import yaml

@Registry
class TritonConfigCreator(Callback):

    def __init__(self, conf_path='', mode=''):

        super().__init__(CallbackOrder.Internal)
        self.mode = mode
        print(self.mode)
        if self.mode == 'multilabel':
            with open('config/classification/multilabel/train_multilabel.yml', encoding="utf8") as f:
                params = yaml.safe_load(f)
                self.aug_path = params['stages']['stage']['data']['transform_path']
                self.num_classes = params['model']['num_classes']
        elif self.mode == 'multiclass':
            with open('config/classification/multiclass/train_multiclass.yml', encoding="utf8") as f:
                params = yaml.safe_load(f)
                self.aug_path = params['stages']['stage']['data']['transform_path']
                self.num_classes = params['model']['num_classes']

        self.conf_path = Path(conf_path)
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.aug_path, encoding="utf8") as f:
                params = yaml.safe_load(f)
                height = params['train']['transforms'][1]['height']
                width = params['train']['transforms'][1]['width']
            
            with open(self.conf_path, "a") as fw:
                fw.write('platform: "onnxruntime_onnx"\n')
                fw.write('input [\n')
                fw.write('{\n')
                fw.write(f'\tname: "input"\n\tdata_type: TYPE_FP32\n\tdims: [-1, 3, {height}, {width}]\n')
                fw.write('}\n]\n')
                fw.write('output [\n')
                fw.write('{\n')
                fw.write(f'\tname: "output"\n\tdata_type: TYPE_FP32\n\tdims: [-1, {self.num_classes}]\n')
                fw.write('}\n]\n')
                fw.write('instance_group [\n')
                fw.write('{\n')
                fw.write('\tcount: 1\n\tkind: KIND_GPU\n\tgpus: [ 0 ]\n}\n')
                fw.write(']\n')
        except AttributeError:
            print('\n'*3)
            print("U've got some misprint( Check that u've written 'multilabel' or 'multiclass')")
            print('\n')