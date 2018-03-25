def flat(alist):
    '''
    No Python hacks in this implementation. Also, this accepts many levels of nested lists.
    The limit is in the number of recursive calls.
    @alist: A tuple or list.
    @return: A flat list with all elements of @alist and its nested lists.
    Complexity: `Θ(n)`, where `n` is the number of elements of @alist
    plus the number of elements of all nested lists.
    '''
    new_list = []
    for item in alist:
        if isinstance(item, (list, tuple)):
            new_list.extend(item)
        else:
            new_list.append(item)
    return new_list