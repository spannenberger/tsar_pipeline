from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
import pandas as pd
from PIL import Image
from tqdm import tqdm


@Registry
class MetricLearningLogger(Callback):
    def __init__(self, logging_incorrect_image_number = 5, logging_uncoordinated_image_number = 5):
        self.logging_incorrect_image_number = logging_incorrect_image_number
        self.logging_uncoordinated_image_number = logging_uncoordinated_image_number
        super().__init__(CallbackOrder.ExternalExtra)


    def on_stage_start(self, state: IRunner):
        """Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа"""

        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')
        mlflow.log_artifact(
            state.hparams['stages']['stage']['data']['transform_path'], 'config/aug_config')
        try:
            mlflow.log_artifact(state.hparams['stages']['stage']['callbacks']['triton']['conf_path'], 'config/triton')
        except FileNotFoundError:
            print('Сant find triton config, because you disabled this callback')
            print('\n'*3)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем в одну папку ошибочные фотографии и фотографии к которым модель отнесла ошибочную,
        в другую папку логаем фотографии которые не прошли по расстоянию
        """

        incorrect_df = pd.read_csv(state.hparams['stages']['stage']['callbacks']['iner']['incorrect_file'], sep=';')
        uncoordinated_df = pd.read_csv(state.hparams['stages']['stage']['callbacks']['iner']['uncoordinated_file'], sep=';')

        incorrect_list = [i for i in incorrect_df['incorrect']]
        couple_list = [i for i in incorrect_df['couple']]
        uncoordinated_list = [i for i in uncoordinated_df['uncoordinated']]

        if self.logging_incorrect_image_number >= len(incorrect_list):
            incorrect_length = len(incorrect_list)
        else:
            incorrect_length = self.logging_incorrect_image_number
        if self.logging_uncoordinated_image_number >= len(uncoordinated_list):
            uncoordinated_length = len(uncoordinated_list)
        else:
            uncoordinated_length = self.logging_uncoordinated_image_number

        for i in tqdm(range(incorrect_length)):
            incorrect_image = Image.open(incorrect_list[i])
            couple_image = Image.open(couple_list[i])
            mlflow.log_image(
                incorrect_image,
                f'incorrect/{incorrect_list[i].split("/")[2]}/{i}/incorrect.png'
            )
            mlflow.log_image(
                couple_image,
                f'incorrect/{incorrect_list[i].split("/")[2]}/{i}/couple.png'
            )
        for i in tqdm(range(uncoordinated_length)):
            uncoordinated_image = Image.open(uncoordinated_list[i])
            mlflow.log_image(
                uncoordinated_image,
                f'uncoordinated/{uncoordinated_list[i].split("/")[2]}/{uncoordinated_list[i].split("/")[3]}.png'
            )
        onnx_checkpoint_names = state.hparams['stages']['stage']['callbacks']['onnx_saver']['checkpoint_names']
        torchsript_checkpoint_names = state.hparams['stages']['stage']['callbacks']['torchscript_saver']['checkpoint_names']

        print('Starting logging convert models... please wait')
        for model in tqdm(torchsript_checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/torchsript/{model}.pt', 'torchscript_models')
            except FileNotFoundError:
                print(f'No such file {model}.pt, nothing to log...')
        
        for model in tqdm(onnx_checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/onnx/{model}.onnx', 'onnx_models')
            except FileNotFoundError:
                print(f'No such file {model}.onnx, nothing to log...')

        mlflow.pytorch.log_model(state.model, artifact_path=state.hparams['model']['_target_'])
        mlflow.end_run()