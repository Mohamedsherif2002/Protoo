from gqa_dataset import GQA
from mnnm_arguments import parse_opt
import os
import pickle
import json


def generate_meta_list(dataset):
    data_list = []
    for idx, entry in enumerate(dataset.data):
        if(idx % 10000 == 0):
            print(f"[{dataset.split}]processing idx {idx} ...", end='\r\n')
        image_id = entry[0]
        questionId = entry[-2]
        data_list.append((image_id, questionId))
    save_p = os.path.join('../mmnm_questions/', 'list_' + dataset.split + ".pkl")
    pickle.dump(data_list, open(save_p, 'wb'))


if __name__ == '__main__':
    args = parse_opt()
    with open('{}/full_vocab.json'.format('../meta_info/'), 'r') as f:
        vocab = json.load(f)

    with open('{}/answer_vocab.json'.format('../meta_info/'), 'r') as f:
        answer = json.load(f)
    basic_kwargs = dict(length=args.length, object_info=os.path.join(args.meta, args.object_info),
                        num_regions=args.num_regions, distribution=args.distribution,
                        vocab=vocab, answer=answer, max_layer=5, num_tokens=args.num_tokens,
                        spatial_info='gqa_spatial_info.json',
                        forbidden=args.forbidden)
    dataset = GQA(split='submission', mode='val', contained_weight=args.contained_weight,
                  threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)
    generate_meta_list(dataset)
