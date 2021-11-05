import json

from utils import calculate_iou, check_results


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """

    data = []
    for bb, sc in zip(predictions['boxes'], predictions['scores']):
        data.append([bb, sc])
    
    # a[start:stop:step] # start through not past stop, by step
    # a[start:stop]  # items start through stop-1
    # a[start:]      # items start through the rest of the array
    # a[:stop]       # items from the beginning through stop-1
    # a[:]           # a copy of the whole array
    # a[::-1]    # all items in the array, reversed
    # a[1::-1]   # the first two items, reversed
    # a[:-3:-1]  # the last two items, reversed
    # a[-3::-1]  # everything except the last two items, reversed
    
    print(sorted(data, key = lambda k: k[1]))
    # [   [[91, 94, 200, 143], 0.06635549149848208],
    #     [[105, 104, 202, 153], 0.07242464990801578],
    #     [[106, 108, 198, 149], 0.07409888441353274],
    #     ...
    #     [[107, 107, 192, 148], 0.814756824762981]]
    # ]
    print(sorted(data, key = lambda k: k[1])[::-1]) # reversed
    # [   [[107, 107, 192, 148], 0.814756824762981]]
    #     ...
    #     [[91, 94, 200, 143], 0.06635549149848208],
    # ]

    data_sorted = sorted(data, key = lambda k: k[1])[::-1]
    filtered = []
    for i, bi in enumerate(data_sorted):
        discard = False
        for j, bj in enumerate(data_sorted):
            if i == j:
                continue
            iou = calculate_iou(bi[0], bj[0])
            if iou > 0.5:
                if bj[1] > bi[1]:
                    discard = True
        if not discard:
            filtered.append(bi)
    return filtered


if __name__ == '__main__':
    with open('./data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = nms(predictions)
    check_results(filtered)