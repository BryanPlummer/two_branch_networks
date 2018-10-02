import torch
import torch.nn as nn
#from feature_extractor import FeatureExtractor

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out),
                         nn.BatchNorm1d(f_out),
                         nn.ReLU(inplace=True),
                         nn.Dropout(p=0.5))

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim, metric_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, metric_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        # L2 normalize each feature vector
        norm = torch.norm(x, p=2, dim=1) + 1e-10
        x = x / norm.expand_as(x)
        return x

class ImageSentenceEmbeddingNetwork(nn.Module):
    def __init__(self, args):
        super(ImageSentenceEmbeddingNetwork, self).__init__()
        #self.featnet = FeatureExtractor(args)
        text_feat_dim = 6000

        if args.resnet:
            image_feat_dim = 2048
        else:
            image_feat_dim = 4096
            
        embedding_dim = args.dim_embed
        metric_dim = args.dim_embed / 4
            
        self.text_branch = EmbedBranch(text_feat_dim, embedding_dim, metric_dim)
        self.image_branch = EmbedBranch(image_feat_dim, embedding_dim, metric_dim)
        if args.cuda:
            self.cuda()
            
    def forward(self, sentences, images):
        #images = self.featnet(images)
        sentences = self.text_branch(sentences)
        images = self.image_branch(images)
        return sentences, images

