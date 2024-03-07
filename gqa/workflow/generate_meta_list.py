from gqa_dataset import GQA
from mnnm_arguments import parse_opt
import os
import pickle
import json


def generate_meta_list(dataset):
    data_list = []
    for idx, entry in enumerate(dataset.data):
        print(f"[{dataset.split}]processing idx {idx} ...", end='\r')
        image_id = entry[0]
        questionId = entry[-2]
        data_list.append((image_id, questionId))
        # Save the data_list to the pickle file
        save_directory = 'mmnm_questions'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        save_p = os.path.join(save_directory, f'list_{dataset.split}.pkl')
        with open(save_p, 'wb') as file:
            pickle.dump(data_list, file)


if __name__ == '__main__':
    args = parse_opt()
    with open('{}/full_vocab.json'.format('../meta_info/'), 'r') as f:
        vocab = json.load(f)

    with open('{}/answer_vocab.json'.format('../meta_info/'), 'r') as f:
        answer = json.load(f)
    basic_kwargs = dict(length=args.length, object_info=os.path.join(args.meta, args.object_info),
                        num_regions=args.num_regions, distribution=args.distribution,
                        vocab=vocab, answer=answer, max_layer=5, num_tokens=args.num_tokens,
                        spatial_info='/kaggle/input/gqa-spatial-features/spatial/gqa_spatial_info.json',
                        forbidden=args.forbidden)
    dataset = GQA(split='submission', mode='val', contained_weight=args.contained_weight,
                  threshold=args.threshold, folder=args.data, cutoff=args.cutoff, **basic_kwargs)
    generate_meta_list(dataset)
