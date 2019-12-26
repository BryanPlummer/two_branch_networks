from __future__ import division
from __future__ import print_function

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy.io import loadmat
import os
import pickle
import numpy as np
from pycocotools.coco import COCO

def load_word_embeddings(word_embedding_filename, embedding_length):
    with open(word_embedding_filename, 'r') as f:
        word_embeddings = {}
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print('Reading word embedding vector %i' % i)
                    
            line = line.strip()
            if not line:
                continue

            vec = line.split()
            if len(vec) != embedding_length + 1:
                continue
            
            label = vec[0].lower()
            vec = np.array([float(x) for x in vec[1:]], np.float32)            
            assert len(vec) == embedding_length
            word_embeddings[label] = vec

    return word_embeddings

def load_coco_captions(args, split):
    stop_words = set(stopwords.words('english'))
    split_fn = os.path.join(args.feat_path, args.dataset, split + '.txt')
    images = [im.strip() for im in open(split_fn, 'r')]
    im2idx = dict(zip(images, range(len(images))))
    images = set(images)

    im2captions = {}
    json = [os.path.join(args.feat_path, args.dataset, 'annotations', 'captions_%s2014.json' % s) for s in ['train', 'val']]
    for fn in json:
        coco = COCO(fn)
        ids = coco.anns.keys()
        for i, ann_id in enumerate(ids):
            im_id = coco.anns[ann_id]['image_id']
            im_id = coco.loadImgs(im_id)[0]['file_name']
            if im_id not in images:
                continue

            caption = str(coco.anns[ann_id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            tokens = [token for token in tokens if token not in stop_words]

            if im_id not in im2captions:
                im2captions[im_id] = []

            im2captions[im_id].append(tokens)

    assert(len(im2idx) == len(im2captions))
    captions = []
    cap2im = []
    for im, idx in im2idx.iteritems():
        im_captions = im2captions[im]
        captions += im_captions
        cap2im.append(np.ones(len(im_captions), np.int64) * idx)

    cap2im = np.hstack(cap2im)
    return captions, cap2im

def load_flickr_captions(args, split):
    stop_words = set(stopwords.words('english'))
    split_fn = os.path.join(args.feat_path, args.dataset, split + '.txt')
    images = [im.strip() for im in open(split_fn, 'r')]
    im2idx = dict(zip(images, range(len(images))))
    images = set(images)
    caption_fn = os.path.join(args.feat_path, args.dataset, 'results_20130124.token')
    im2captions = {}
    with open(caption_fn, 'r') as f:
        for line in f:
            line = line.strip().lower().split()
            im = line[0].split('.')[0]
            if im in images:
                if im not in im2captions:
                    im2captions[im] = []

                im2captions[im].append([token for token in line[1:] if token not in stop_words])

    assert(len(im2idx) == len(im2captions))
    captions = []
    cap2im = []
    for im, idx in im2idx.iteritems():
        im_captions = im2captions[im]
        captions += im_captions
        cap2im.append(np.ones(len(im_captions), np.int32) * idx)

    cap2im = np.hstack(cap2im)
    return captions, cap2im

class DatasetLoader:
    """ Dataset loader class that loads feature matrices from given paths and
        create shuffled batch for training, unshuffled batch for evaluation.
    """
    def __init__(self, args, split='train'):
        feat_path = os.path.join(args.feat_path, args.dataset, split + '_features.npy')
        print('Loading features from', feat_path)
        self.im_feats = np.load(feat_path)
        if args.dataset == 'flickr':
            self.captions, self.cap2im = load_flickr_captions(args, split)
        else:
            self.captions, self.cap2im = load_coco_captions(args, split)

            if split == 'val':
                # let's only take the first 1K images for MSCOCO images
                num_images = 1000
                self.im_feats = self.im_feats[:num_images]
                subset_ims = self.cap2im < num_images
                self.captions = [caption for caption, is_val in zip(self.captions, subset_ims) if is_val]
                self.cap2im = [im for im, is_val in zip(self.cap2im, subset_ims) if is_val]

        assert len(self.cap2im) == len(self.captions)
        if split != 'train':
            self.labels = np.zeros((len(self.cap2im), len(self.im_feats)), np.float)
            self.labels[(range(len(self.cap2im)), self.cap2im)] = 1
        else:
            self.im2cap = {}
            for cap, im in enumerate(self.cap2im):
                if im not in self.im2cap:
                    self.im2cap[im] = []

                self.im2cap[im].append(cap)

        print('Loading complete')
        self.split = split
        self.sample_size = args.sample_size
        self.sent_inds = range(len(self.captions)) # we will shuffle this every epoch for training

    def build_vocab(self, cache_filename, word_embeddings_filename = None, embedding_length = 300):
        if os.path.exists(cache_filename):
            vocab_data = pickle.load(open(cache_filename, 'rb'))
            self.max_length = vocab_data['max_length']
            self.tok2idx = vocab_data['tok2idx']
            vecs = vocab_data['vecs']
        else:
            assert word_embeddings_filename is not None
            word_embeddings = load_word_embeddings(word_embeddings_filename, embedding_length)
            self.max_length = 0
            vocab = set()
            for caption in self.captions:
                tokens = [token for token in caption if token in word_embeddings]
                vocab.update(tokens)
                self.max_length = max(self.max_length, len(tokens))

            vocab = list(vocab)
            # +1 for a padding vector which *must* be the 0th index
            self.tok2idx = dict(zip(vocab, range(1, len(vocab) + 1)))
            vecs = np.zeros((len(vocab) + 1, embedding_length), np.float32)
            for i, token in enumerate(vocab):
                vecs[i + 1] = word_embeddings[token]
            
            vocab_data = {'max_length' : self.max_length,
                          'tok2idx' : self.tok2idx,
                          'vecs' : vecs}

            pickle.dump(vocab_data, open(cache_filename, 'wb'))

        self.sent_feats = np.zeros((len(self.captions), self.max_length), np.int64)
        for i, caption in enumerate(self.captions):
            tokens = [self.tok2idx[token] for token in caption if token in self.tok2idx]
            self.sent_feats[i, :len(tokens)] = tokens


        self.sent_feat_shape = self.sent_feats.shape
        return vecs

    def __len__(self):
        return len(self.captions)

    def shuffle_inds(self):
        '''
        shuffle the indices in training (run this once per epoch)
        nop for testing and validation
        '''
        if self.split == 'train':
            np.random.shuffle(self.sent_inds)

    def __getitem__(self, index):
        im = self.cap2im[index]
        im_feat = self.im_feats[self.cap2im[index]]
        sample_index = np.random.choice(
            [i for i in self.im2cap[im] if i != index],
            self.sample_size - 1, replace=False)
        sample_index = sorted(np.append(sample_index, index))
        sent_feat = self.sent_feats[sample_index]
        return im_feat, sent_feat

    def sample_items(self, sample_inds, sample_size):
        '''
        for each index, return the  relevant image and sentence features
        sample_inds: a list of sent indices
        sample_size: number of neighbor sentences to sample per index.
        '''
        im_feats_b = self.im_feats[self.cap2im[sample_inds],:]
        sent_feats_b = []
        for ind, im in zip(sample_inds, self.cap2im[sample_inds]):
            # ind is an index for sentence
            sample_index = np.random.choice(
                    [i for i in self.im2cap[im] if i != ind],
                    sample_size - 1, replace=False)
            sample_index = sorted(np.append(sample_index, ind))
            sent_feats_b.append(self.sent_feats[sample_index])
        sent_feats_b = np.concatenate(sent_feats_b, axis=0)
        return (im_feats_b, sent_feats_b)

    def get_batch(self, batch_index, batch_size, sample_size):
        start_ind = batch_index * batch_size
        end_ind = start_ind + batch_size
        sample_inds = self.sent_inds[start_ind : end_ind]
        (im_feats, sent_feats) = self.sample_items(sample_inds, sample_size)
        # Each row of the labels is the label for one sentence,
        # with corresponding image index sent to True.
        labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        return(im_feats, sent_feats, labels)
