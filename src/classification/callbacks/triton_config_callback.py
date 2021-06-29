from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
from google.protobuf import text_format
from pprint import pprint
import yaml

@Registry
class TritonConfigCreator(Callback):

    def __init__(self, conf_path='', count=0, kind='', gpus=[], mode=''):

        super().__init__(CallbackOrder.Internal)
        self.mode = mode
        self.count = count
        self.kind = kind
        self.gpus = gpus
        print(self.mode)
        if self.mode == 'multilabel':
            with open('config/classification/multilabel/train_multilabel.yml', encoding="utf8") as config_yaml:
                params = yaml.safe_load(config_yaml)
                self.aug_path = params['stages']['stage']['data']['transform_path']
                self.output_size = params['model']['num_classes']
        elif self.mode == 'multiclass':
            with open('config/classification/multiclass/train_multiclass.yml', encoding="utf8") as config_yaml:
                params = yaml.safe_load(config_yaml)
                self.aug_path = params['stages']['stage']['data']['transform_path']
                self.output_size = params['model']['num_classes']
        elif self.mode == 'metric_learning':
            with open('config/metric_learning/train_metric_learning.yml', encoding="utf8") as config_yaml:
                params = yaml.safe_load(config_yaml)
                self.aug_path = params['stages']['stage']['data']['transform_path']
                self.output_size = params['stages']['stage']['callbacks']['criterion']['embeding_size']

        self.conf_path = Path(conf_path)
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.aug_path, encoding="utf8") as aug_yaml:
                params = yaml.safe_load(aug_yaml)
                height = params['train']['transforms'][1]['height']
                width = params['train']['transforms'][1]['width']
            
            with open(self.conf_path, "a") as triton_config:
                triton_config.write('platform: "onnxruntime_onnx"\n')
                triton_config.write('input [\n')
                triton_config.write('{\n')
                triton_config.write(f'\tname: "input"\n\tdata_type: TYPE_FP32\n\tdims: [-1, 3, {height}, {width}]\n')
                triton_config.write('}\n]\n')
                triton_config.write('output [\n')
                triton_config.write('{\n')
                triton_config.write(f'\tname: "output"\n\tdata_type: TYPE_FP32\n\tdims: [-1, {self.output_size}]\n')
                triton_config.write('}\n]\n')
                triton_config.write('instance_group [\n')
                triton_config.write('{\n')
                if self.count != 'None':
                    triton_config.write(f'\tcount: {self.count}\n\t')
                else: pass
                if self.kind != 'None':
                    triton_config.write(f'kind: {self.kind}\n\t')
                else: pass
                if self.gpus != 'None':
                    triton_config.write(f'gpus: {self.gpus}\n')
                else: pass
                triton_config.write('}\n')
                triton_config.write(']\n')
        except AttributeError:
            print('\n'*3)
            print("U've got some misprint( Check that u've written 'multilabel' or 'multiclass')")
            print('\n')