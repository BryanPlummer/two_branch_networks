import pickle
import numbers
import torch
import torch.nn as nn

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
        x = nn.functional.normalize(x)
        return x

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = torch.sum(x1*x1, 1).view(-1, 1)
    x2_square = torch.sum(x2*x2, 1).view(1, -1)
    return torch.sqrt(x1_square - 2 * torch.mm(x1, x2.transpose(0, 1)) + x2_square + 1e-4)

def embedding_loss(im_embeds, sent_embeds, im_labels, args):
    """
        im_embeds: (b, 512) image embedding tensors
        sent_embeds: (sample_size * b, 512) sentence embedding tensors
            where the order of sentence corresponds to the order of images and
            setnteces for the same image are next to each other
        im_labels: (sample_size * b, b) boolean tensor, where (i, j) entry is
            True if and only if sentence[i], image[j] is a positive pair
    """
    # compute embedding loss
    sent_im_ratio = args.sample_size
    num_img = im_embeds.size(0)
    num_sent = num_img * sent_im_ratio

    sent_im_dist = pdist(sent_embeds, im_embeds)
    im_labels = im_labels > 0

    # image loss: sentence, positive image, and negative image
    pos_pair_dist = torch.masked_select(sent_im_dist, im_labels).view(num_sent, 1)
    neg_pair_dist = torch.masked_select(sent_im_dist, ~im_labels).view(num_sent, -1)
    im_loss = torch.clamp(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = im_loss.topk(args.num_neg_sample)[0].mean()
 
    # sentence loss: image, positive sentence, and negative sentence
    neg_pair_dist = torch.masked_select(sent_im_dist.t(), ~im_labels.t()).view(num_img, -1)
    neg_pair_dist = neg_pair_dist.repeat(1, sent_im_ratio).view(num_sent, -1)
    sent_loss = torch.clamp(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = sent_loss.topk(args.num_neg_sample)[0].mean()

    # sentence only loss (neighborhood-preserving constraints)
    sent_sent_dist = pdist(sent_embeds, sent_embeds)
    sent_sent_mask = im_labels.t().repeat(1, sent_im_ratio).view(num_sent, num_sent)
    pos_pair_dist = torch.masked_select(sent_sent_dist, sent_sent_mask).view(-1, sent_im_ratio)
    pos_pair_dist = pos_pair_dist.max(dim=1, keepdim=True)[0]
    neg_pair_dist = torch.masked_select(sent_sent_dist, ~sent_sent_mask).view(num_sent, -1)
    sent_only_loss = torch.clamp(args.margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_only_loss = sent_only_loss.topk(args.num_neg_sample)[0].mean()

    loss = im_loss * args.im_loss_factor + sent_loss + sent_only_loss * args.sent_only_loss_factor
    return loss

class ImageSentenceEmbeddingNetwork(nn.Module):
    def __init__(self, args, vecs, image_feature_dim):
        super(ImageSentenceEmbeddingNetwork, self).__init__()
        embedding_dim = args.dim_embed
        metric_dim = args.dim_embed / 4
        n_tokens, token_dim = vecs.shape
        self.words = nn.Embedding(n_tokens, token_dim)
        self.words.weight = nn.Parameter(torch.from_numpy(vecs))
        self.vecs = torch.from_numpy(vecs)
        self.word_reg = nn.MSELoss()
        if args.language_model == 'attend':
            self.word_attention = nn.Sequential(nn.Linear(vecs.shape[1] * 2, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Softmax(dim=1))

        self.text_branch = EmbedBranch(token_dim, embedding_dim, metric_dim)
        self.image_branch = EmbedBranch(image_feature_dim, embedding_dim, metric_dim)
        self.args = args
        if args.cuda:
            self.cuda()
            self.vecs = self.vecs.cuda()


    def forward(self, images, tokens):
        words = self.words(tokens)
        n_words = torch.sum(tokens > 0, 1).float() + 1e-10
        sum_words = words.sum(1).squeeze()
        sentences = sum_words / n_words.unsqueeze(1)
        
        if self.args.language_model == 'attend':
            max_length = tokens.size(-1)
            context_vector = sentences.unsqueeze(1).repeat(1, max_length, 1)
            attention_inputs = torch.cat((context_vector, words), 2)
            attention_weights = self.word_attention(attention_inputs)
            sentences = nn.functional.normalize(torch.sum(words * attention_weights, 1))

        sentences = self.text_branch(sentences)
        images = self.image_branch(images)
        return images, sentences

    def train_forward(self, images, sentences, im_labels):
        im_embeds, sent_embeds = self(images, sentences)
        embed_loss = embedding_loss(im_embeds, sent_embeds, im_labels, self.args)
        word_loss = self.word_reg(self.words.weight, self.vecs)
        loss = embed_loss + word_loss * self.args.word_embedding_reg
        return loss, im_embeds, sent_embeds
