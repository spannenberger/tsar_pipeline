
def get_from_dict(dictionary, path, default=None):
    path = path.split(':')
    import pdb
    pdb.set_trace()
    for key in path:
        try:
            dictionary = dictionary[key]
        except KeyError as e:
            raise KeyError(
                f"config.yaml has not key '{path}' key '{e.args[0]}' is wrong.")
    return dictionary
