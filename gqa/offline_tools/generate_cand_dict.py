# generate candidate answers given programs
import time




from nltk.tokenize import word_tokenize
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import copy
import os
import random

PAD = 0
EOS = 1
UNK = 2
SOS = 3
T = 1
NUM_BBOX = 36
VISUAL_FEAT = 2048
BIAS = 1000
EMB_DIM = 100

OBJECT_FUNCS = ['relate', 'relate_inv', 'relate_name', 'relate_inv_name', 'select', 'relate_attr', 'filter', 'filter_not', 'filter_h']
STRING_FUNCS = ['query_n', 'query_h', 'query', 'query_f', 'choose_n', 'choose_f', 'choose', 'choose_attr', 'choose_h', 'choose_v', 'choose_rel_inv', 'choose_subj', 'common']
BINARY_FUNCS = ['verify', 'verify_f', 'verify_h', 'verify_v', 'verify_rel', 'verify_rel_inv', 'exist', 'or', 'and', 'different', 'same', 'same_attr', 'different_attr']

BBOX_ONTOLOGY = {'darkness': ['dark', 'bright'],
				 'dryness': ['wet', 'dry'],
				 'colorful': ['colorful', 'shiny'],
				 'leaf': ['leafy', 'bare'],
				 'emotion': ['happy', 'calm'],
				 'sports': ['baseball', 'tennis'],
				 'flatness': ['flat', 'curved'],
				 'lightness': ['light', 'heavy'],
				 'gender': ['male', 'female'],
				 'width': ['wide', 'narrow'],
				 'depth': ['deep', 'shallow'],
				 'hardness': ['hard', 'soft'],
				 'cleanliness': ['clean', 'dirty'],
				 'switch': ['on', 'off'],
				 'thickness': ['thin', 'thick'],
				 'openness': ['open', 'closed'],
				 'height': ['tall', 'short'],
				 'length': ['long', 'short'],
				 'fullness': ['full', 'empty'],
				 'age': ['young', 'old'],
				 'size': ['large', 'small'],
				 'pattern': ['checkered', 'striped', 'dress', 'dotted'],
				 'shape': ['round', 'rectangular', 'triangular', 'square'],
				 'activity': ['waiting', 'staring', 'drinking', 'playing', 'eating', 'cooking', 'resting', 'sleeping', 'posing', 'talking',
							  'looking down', 'looking up', 'driving', 'reading', 'brushing teeth', 'flying', 'surfing', 'skiing', 'hanging'],
				 'pose': ['walking', 'standing', 'lying', 'sitting', 'running', 'jumping', 'crouching', 'bending', 'smiling', 'grazing'],
				 'material': ['wood', 'plastic', 'metal', 'glass', 'leather', 'leather', 'porcelain', 'concrete', 'paper', 'stone', 'brick'],
				 'color': ['white', 'red', 'black', 'green', 'silver', 'gold', 'khaki', 'gray', 'dark', 'pink', 'dark blue', 'dark brown',
						   'blue', 'yellow', 'tan', 'brown', 'orange', 'purple', 'beige', 'blond', 'brunette', 'maroon', 'light blue', 'light brown']}

SCENE_ONTOLOGY = {'location': ['indoors', 'outdoors'],
				 'weather': ['clear', 'overcast', 'cloudless', 'cloudy', 'sunny', 'foggy', 'rainy'],
				 'room': ['bedroom', 'kitchen', 'bathroom', 'living room'],
				 'place': ['road', 'sidewalk', 'field', 'beach', 'park', 'grass', 'farm', 'ocean', 'pavement',
						   'lake', 'street', 'train station', 'hotel room', 'church', 'restaurant', 'forest', 'path',
						   'display', 'store', 'river', 'sea', 'yard', 'airport', 'parking lot']}

ONTOLOGY = copy.deepcopy(BBOX_ONTOLOGY)
ONTOLOGY.update(SCENE_ONTOLOGY)

BBOX_ATTR = list(BBOX_ONTOLOGY.keys())

SCENE_ATTR = list(SCENE_ONTOLOGY.keys())

BBOX_ATTRIBUTES = {}
for k, v in BBOX_ONTOLOGY.items():
	for i, item in enumerate(v):
		if item in BBOX_ATTRIBUTES:
			BBOX_ATTRIBUTES[item].append((BBOX_ATTR.index(k), i))
		else:
			BBOX_ATTRIBUTES[item] = [(BBOX_ATTR.index(k), i)]

SCENE_ATTRIBUTES = {}
for k, v in SCENE_ONTOLOGY.items():
	for i, item in enumerate(v):
		if item in SCENE_ATTRIBUTES:
			SCENE_ATTRIBUTES[item].append((SCENE_ATTR.index(k), i))
		else:
			SCENE_ATTRIBUTES[item] = [(SCENE_ATTR.index(k), i)]

with open('meta_info/GQA_hypernym.json') as f:
	hypernym = json.load(f)

with open('meta_info/objects.json') as f:
	OBJECTS_INV = json.load(f)
	OBJECTS = {k:i for i, k in enumerate(OBJECTS_INV)}

with open('meta_info/predicates.json') as f:
	RELATIONS_INV = json.load(f)
	RELATIONS = {k:i for i, k in enumerate(RELATIONS_INV)}

with open('meta_info/attributes.json') as f:
	ATTRIBUTES_INV = json.load(f)
	ATTRIBUTES = {k:i for i, k in enumerate(ATTRIBUTES_INV)}

with open('meta_info/obj2attribute.json') as f:
	mapping = json.load(f)

OBJ2ATTRIBUTES = {}
for k, vs in mapping.items():
	tmp = set()
	for v in vs:
		if v in BBOX_ATTRIBUTES:
			for attr_k, _ in BBOX_ATTRIBUTES[v]:
				tmp.add(attr_k)
	OBJ2ATTRIBUTES[k] = list(tmp)


def show_im(k, x, y, w, h, title):
	im = np.array(Image.open("../images/{}.jpg".format(k)), dtype=np.uint8)
	# Create figure and axes
	fig, ax = plt.subplots(1)
	ax.imshow(im)
	# Create a Rectangle patch
	rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
	# Add the patch to the Axes
	ax.set_title(title)
	ax.add_patch(rect)

	plt.show()


def show_im_bboxes(k, coordinates):
	im = np.array(Image.open("../images/{}.jpg".format(k)), dtype=np.uint8)
	height = im.shape[0]
	width = im.shape[1]
	# Create figure and axes
	fig, ax = plt.subplots(1)
	ax.imshow(im)
	colors = ['red', 'yellow', 'black', 'blue', 'orange', 'grey', 'cyan', 'green', 'purple']
	# Create a Rectangle patch
	for coordinate in coordinates:
		x, y = coordinate[0] * width, coordinate[1] * height
		w, h = (coordinate[2] - coordinate[0]) * width, (coordinate[3] - coordinate[1]) * height
		color = random.choice(colors)
		rect1 = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
		ax.add_patch(rect1)

	plt.show()


def intersect(bbox1, bbox2, contained=False, option="xywh"):
	if option == 'xywh':
		x_inter = max(bbox1[0], bbox2[0])
		y_inter = max(bbox1[1], bbox2[1])
		x_p_inter = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
		y_p_inter = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
		intersect_area = max(x_p_inter - x_inter, 0) * max(y_p_inter - y_inter, 0)
		whole = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersect_area
	elif option == 'x1y1x2y2':
		x_inter = max(bbox1[0], bbox2[0])
		y_inter = max(bbox1[1], bbox2[1])
		x_p_inter = min(bbox1[2], bbox2[2])
		y_p_inter = min(bbox1[3], bbox2[3])
		intersect_area = max(x_p_inter - x_inter, 0) * max(y_p_inter - y_inter, 0)
		whole = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - intersect_area
	else:
		raise NotImplementedError

	if contained:
		return intersect_area / (whole + 0.01), intersect_area / (bbox1[2] * bbox1[3] + 0.01)
	else:
		return intersect_area / (whole + 0.01)


def parse_program(string):
	if '=' in string:
		result, function = string.split('=')
	else:
		function = string
		result = "?"

	func, arguments = function.split('(')
	if len(arguments) == 1:
		return result, func, []
	else:
		arguments = list(map(lambda x:x.strip(), arguments[:-1].split(',')))
		return result, func, arguments
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
import time
from torch import nn


class GQA(Dataset):
    def __init__(self, **args):
        self.mode = args['mode']
        self.split = args['split']
        self.vocab = args['vocab']
        self.answer_vocab = args['answer']
        self.num_tokens = args['num_tokens']
        self.num_regions = args['num_regions']
        self.LENGTH = args['length']
        self.MAX_LAYER = args['max_layer']
        self.folder = args['folder']
        self.threshold = args['threshold']
        self.contained_weight = args['contained_weight']
        self.cutoff = args['cutoff']
        self.distribution = args['distribution']
        self.failure_p = args['failure_path'] if 'failure_path' in args else None

        if args['forbidden'] != '':
            with open(args['forbidden'], 'r') as f:
                self.forbidden = json.load(f)
            self.forbidden = set(self.forbidden)
        else:
            self.forbidden = set([])
        if self.failure_p is not None:
            print(f"Loading failed data from {self.failure_p}.")
            self.data = pickle.load(open(self.failure_p, 'rb'))
        else:
            meta_list_p = os.path.join('mmnm_questions/', 'list_' + self.split + ".pkl")
            print(f"Loading meta data from {meta_list_p}.")
            self.data = pickle.load(open(meta_list_p, 'rb'))

        with open(args['object_info']) as f:
            self.object_info = json.load(f)
        print(f"there are in total {len(self.data)} instances.")

    def __getitem__(self, index):
        if self.failure_p:
            question_id, image_id = self.data[index]
        else:
            image_id, question_id = self.data[index]
        cur_p = os.path.join('mmnm_questions/', 'mmnm_{}.pkl'.format(image_id))
        entry = pickle.load(open(cur_p, 'rb'))[question_id]
        obj_info = self.object_info[entry[0]]
        if not entry[0].startswith('n'):
            if len(entry[0]) < 7:
                entry[0] = "0" * (7 - len(entry[0])) + entry[0]

        image_id = entry[0]
        question = entry[1]
        inputs = entry[3]
        prog_key = inputs[-1][0]
        connection = entry[4]
        length = min(len(inputs), self.LENGTH)

        # Prepare Question
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, UNK) for _ in idxs]
        question += [PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')

        question_masks = np.zeros((len(question),), 'float32')
        question_masks[:len(idxs)] = 1.

        # Prepare Program
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH,), 'int64')
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, UNK)

        # Prepare Program mask
        program_masks = np.zeros((self.LENGTH,), 'float32')
        program_masks[:length] = 1.

        # Prepare Program Transition Mask
        transition_masks = np.zeros((self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32')
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
                    activate_mask[i][idx] = 1
            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass
        vis_mask = np.zeros((self.num_regions,), 'float32')
        # new_bottom_up = self.all_bottom_up[image_id]

        # Prepare Vision Feature
        bottom_up = np.load(os.path.join(self.folder, 'gqa_{}.npz'.format(image_id)))
        adaptive_num_regions = min((bottom_up['conf'] > self.threshold).sum(), self.num_regions)

        # Cut off the bottom up features
        object_feat = bottom_up['features'][:adaptive_num_regions]
        bbox_feat = bottom_up['norm_bb'][:adaptive_num_regions]
        vis_mask[:bbox_feat.shape[0]] = 1.

        # Padding zero
        if object_feat.shape[0] < self.num_regions:
            padding = self.num_regions - object_feat.shape[0]
            object_feat = np.concatenate([object_feat, np.zeros(
                (padding, object_feat.shape[1]), 'float32')], 0)
        if bbox_feat.shape[0] < self.num_regions:
            padding = self.num_regions - bbox_feat.shape[0]
            bbox_feat = np.concatenate([bbox_feat, np.zeros(
                (padding, bbox_feat.shape[1]), 'float32')], 0)
        num_regions = bbox_feat.shape[0]

        # exist = np.full((self.LENGTH, ), -1, 'float32')
        returns = entry[2]
        intermediate_idx = np.full(
            (self.LENGTH, num_regions + 1), 0, 'float32')
        intersect_iou = np.full(
            (length - 1, num_regions + 1), 0., 'float32')
        if self.mode == 'train':
            for idx in range(length - 1):
                if isinstance(returns[idx], list):
                    if returns[idx] == [-1, -1, -1, -1]:
                        intermediate_idx[idx][num_regions] = 1
                    else:
                        gt_coordinate = (returns[idx][0] / (obj_info['width'] + 0.),
                                         returns[idx][1] / (obj_info['height'] + 0.),
                                         (returns[idx][2] + returns[idx][0]) / (obj_info['width'] + 0.),
                                         (returns[idx][3] + returns[idx][1]) / (obj_info['height'] + 0.))
                        for i in range(num_regions):
                            intersected, contain = intersect(gt_coordinate, bbox_feat[i, :4], True, 'x1y1x2y2')
                            intersect_iou[idx][i] = intersected  # + self.contained_weight * contain

                        # if self.distribution:
                        # mask = (intersect_iou[idx] > self.cutoff).astype('float32')
                        # intersect_iou[idx] *= mask
                        intermediate_idx[idx] = intersect_iou[idx] / (intersect_iou[idx].sum() + 0.001)
                        # else:
                        #    intermediate_idx[idx] = (intersect_iou[idx] > self.cutoff).astype('float32')
                        #    intermediate_idx[idx] = intermediate_idx[idx] / (intermediate_idx[idx].sum() + 0.001)
        else:
            intermediate_idx = 0
        # Prepare index selection
        index = length - 1
        # Prepare answer
        answer_id = self.answer_vocab.get(entry[-1], UNK)

        return question, question_masks, program, program_masks, transition_masks, activate_mask, object_feat, \
               bbox_feat, vis_mask, index, depth, intermediate_idx, answer_id, question_id, image_id

    def __len__(self):
        return len(self.data)


def create_splited_questions(dataset, save_dir='mmnm_questions/'):
    for idx, entry in enumerate(dataset.data):
        print(f"[{dataset.mode}]processing idx {idx} ...", end='\r')
        image_id = entry[0]
        questionId = entry[-2]
        save_p = os.path.join(save_dir, 'mmnm_{}.pkl'.format(image_id))
        if os.path.exists(save_p):
            cur_meta = pickle.load(open(save_p, 'rb'))
        else:
            cur_meta = {}
        cur_meta[questionId] = entry
        pickle.dump(cur_meta, open(save_p, 'wb'))


def generate_meta_list(dataset):
    data_list = []
    for idx, entry in enumerate(dataset.data):
        print(f"[{dataset.split}]processing idx {idx} ...", end='\r')
        image_id = entry[0]
        questionId = entry[-2]
        data_list.append((image_id, questionId))
    save_p = os.path.join('mmnm_questions/', 'list_' + dataset.split + ".pkl")
    pickle.dump(data_list, open(save_p, 'wb'))


def test_dataset():
    with open('{}/full_vocab.json'.format('meta_info/'), 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('{}/answer_vocab.json'.format('meta_info/'), 'r') as f:
        answer = json.load(f)
        inv_answer = {v: k for k, v in answer.items()}

    train_dataset = GQA(split='trainval_all_fully', mode='train', contained_weight=0.1,
                        threshold=0.0, folder='gqa_bottom_up_features/', cutoff=0.5, vocab=vocab, answer=answer,
                        forbidden='', object_info='meta_info/gqa_objects_merged_info.json', num_tokens=30,
                        num_regions=48, length=9, max_layer=5, distribution=False)
    test_d = train_dataset[0]
    print(test_d)














def generate_dicts(encode=True):
    with open('{}/full_vocab.json'.format('meta_info/'), 'r') as f:
        vocab = json.load(f)

    with open('{}/answer_vocab.json'.format('meta_info/'), 'r') as f:
        answer = json.load(f)

    split = 'trainval_all_fully'
    mode = 'train'
    gqa_d = GQA(split=split, mode=mode, contained_weight=0.1, threshold=0.0, folder='gqa_bottom_up_features/',
                cutoff=0.5, vocab=vocab, answer=answer, forbidden='', object_info='meta_info/gqa_objects_merged_info.json',
                num_tokens=30, num_regions=48, length=9, max_layer=5, distribution=False, failure_path=None)

    type2cand_dict = {}
    start_t = time.time()
    for idx, ele in enumerate(gqa_d.data):
        if idx % 1000 == 0:
            time_per_iter = (time.time() - start_t) / (idx + 1e-9)
            print(f"{idx} / {len(gqa_d.data)}, finished. Time per iter: {time_per_iter:.3f}.", end='\r')
            type2cand_dict_p = os.path.join('meta_info/type2cand_dict.pkl')
            pickle.dump(type2cand_dict, open(type2cand_dict_p, 'wb'))
        image_id, question_id = ele[0], ele[1]
        cur_p = os.path.join('mmnm_questions/', 'mmnm_{}.pkl'.format(image_id))
        entry = pickle.load(open(cur_p, 'rb'))[question_id]
        # prog_type = [ele for ele in entry[3][-1] if ele is not None]
        # prog_type = '_'.join(prog_type)
        prog_type = entry[3][-1][0]
        answer = entry[-1]
        if encode:
            answer = gqa_d.answer_vocab.get(answer, UNK)
        if prog_type not in type2cand_dict:
            type2cand_dict[prog_type] = set()
        if answer not in type2cand_dict[prog_type]:
            type2cand_dict[prog_type].add(answer)


if __name__ == '__main__':
    generate_dicts()
