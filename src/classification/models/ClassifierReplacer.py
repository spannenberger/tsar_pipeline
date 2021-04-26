from torch import nn

def replace_classifier(model, num_classes):
    classifier_name, old_classifier = model._modules.popitem()
    if isinstance(old_classifier, nn.Sequential):
        input_shape = old_classifier[-1].in_features
        old_classifier[-1] = nn.Linear(input_shape, num_classes)

    elif isinstance(old_classifier, nn.Linear):
        input_shape = old_classifier.in_features
        old_classifier = nn.Linear(input_shape, num_classes)
    else:
        raise Exception("Uknown type of classifier {}".format(type(old_classifier)))
    model.add_module(classifier_name, old_classifier)