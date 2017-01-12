# select subsets of a list of named tuples
def argwhere(iterable, **conditions):
    idx = []
    for i, element in enumerate(iterable):
        if all([element._asdict()[k] is v for k,v in conditions.items()]):
            idx.append(i)
    return idx


def where(iterable, **conditions):
    elements = []
    for i, element in enumerate(iterable):
        if all([element._asdict()[k] is v for k,v in conditions.items()]):
            elements.append(element)
    return elements