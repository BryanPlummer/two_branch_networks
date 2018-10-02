import numpy as np
from scipy.io import loadmat
import pdb
import h5py

class ImageSentenceFeatures:
    def __init__(self, args, split):
        # code only supports sampling 1 extra sentence, or take them all
        if split == 'train':
            sample_k_sentences = 1
        else:
            sample_k_sentences = 5
        
        sentence_feats = h5py.File('data/flickr_%s_whole_sentence.mat' % split)
        self.sentences = np.transpose(sentence_feats['text_features']).astype(np.float32)
        if args.resnet:
            images = loadmat('data/flickr_resnet_%s.mat' % split)
        else:
            images = loadmat('data/flickr_vgg19_%s.mat' % split)

        self.images = images['image_features'].astype(np.float32)
        self.sample_k_sentences = sample_k_sentences
        self.split = split

    def __len__(self):
        return self.sentences.shape[0]/self.sample_k_sentences

    def __getitem__(self, index):
        # code assumes there are five sentences per image
        if self.sample_k_sentences != 5:
            im_idx = int(np.floor(index / 5.0))
            sent_range = range(im_idx*5,(im_idx+1)*5)
            sentence = self.sentences[index,:]
            sent_range.remove(index)
            sentence_pair = np.random.choice(sent_range)
            sentence = np.hstack((sentence, self.sentences[sentence_pair,:]))
        else:
            im_idx = index
            sent_range = range(im_idx*5,(im_idx+1)*5)
            sentence = np.hstack([self.sentences[i,:] for i in sent_range])

        img = self.images[im_idx,:]
        return img, sentence, im_idx
        
        
