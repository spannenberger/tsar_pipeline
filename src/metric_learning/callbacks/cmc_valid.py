from catalyst import dl


class CustomCMC(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=dl.CMCScoreCallback(*args, **kwargs), loaders=loaders)


# data = {'embeddings_key': 'embeddings', 'labels_key': 'targets',
#         'is_query_key': 'is_query', 'topk_args': [1], 'loaders': 'valid'}
# a = CustomCMS(**data)
# print(a)
