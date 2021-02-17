from catalyst import dl
import torch
class TTASupervisedRunner(dl.SupervisedRunner):

    def _run_experiment(self, *args,**kwargs):

        try:
            self.tta = self.experiment.hparams['stages']['data_params']['tta']
        except Exception as e:
            self.tta = 1

        super()._run_experiment(*args,**kwargs)


    def _run_batch(self):
        
        self.input = self._handle_device(batch=self.input)
        self._run_event("on_batch_start")
        self._handle_batch(batch=self.input)
        #print(dir(self.experiment.get_loaders(stage=self.stage)['train']))
        if self.is_valid_loader:
            # resize inputs
            self.input['image_name'] = [x for x in self.input['image_name'][::self.tta]]
            self.input['targets']= torch.stack([x for x in self.input['targets'][::self.tta]])
            # averaging outputs
            logits = self.output['logits']
            self.output['logits'] = torch.stack([logits[i: i + self.tta].mean(dim = 0) for i in range(0, len(logits) - self.tta + 1, self.tta)])
        self._run_event("on_batch_end")