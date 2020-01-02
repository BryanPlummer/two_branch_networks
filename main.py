from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np
import shutil
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from dataset_utils import DatasetLoader
from retrieval_model import ImageSentenceEmbeddingNetwork, pdist

def main():
    # Load data.
    train_loader = DatasetLoader(FLAGS, 'train')

    vocab_filename = os.path.join(FLAGS.feat_path, FLAGS.dataset, 'vocab.pkl')
    word_embedding_filename = os.path.join(FLAGS.feat_path, 'mt_grovle.txt')
    embedding_length = 300
    print('Loading vocab')
    vecs = train_loader.build_vocab(vocab_filename, word_embedding_filename, embedding_length)
    print('Loading complete')

    kwargs = {'num_workers': 8, 'pin_memory': True} if FLAGS.cuda else {}
    train_loader = torch.utils.data.DataLoader(train_loader,
            batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    test_loader = DatasetLoader(FLAGS, 'test')
    val_loader = DatasetLoader(FLAGS, 'val')

    # Assumes the train_loader has already built the vocab and can be loaded
    # from the cached file.
    test_loader.build_vocab(vocab_filename)
    val_loader.build_vocab(vocab_filename)

    image_feature_dim = train_loader.dataset.im_feats.shape[-1]
    model = ImageSentenceEmbeddingNetwork(FLAGS, vecs, image_feature_dim)

     # optionally resume from a checkpoint
    start_epoch, best_acc = load_checkpoint(model, FLAGS.resume)
    cudnn.benchmark = True

    if FLAGS.test:
        test_acc = test(test_loader, model)
        sys.exit()

    parameters = [{'params' : model.words.parameters(), 'weight_decay' : 0.},
                  {'params' : model.image_branch.fc1.parameters()},
                  {'params' : model.image_branch.fc2.parameters()},
                  {'params' : model.text_branch.fc1.parameters(), 'lr' : FLAGS.lr*FLAGS.text_lr_multi},
                  {'params' : model.text_branch.fc2.parameters(), 'lr' : FLAGS.lr*FLAGS.text_lr_multi}]

    if FLAGS.language_model == 'attend':
        parameters.append({'params' : model.word_attention.parameters()})

    optimizer = optim.Adam(parameters, lr=FLAGS.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.794)
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    save_directory = os.path.join(FLAGS.save_dir, FLAGS.dataset, FLAGS.name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    epoch = start_epoch
    best_epoch = epoch
    while (epoch - best_epoch) < FLAGS.no_gain_stop and (FLAGS.max_num_epoch < 0 or epoch <= FLAGS.max_num_epoch):
        # train for one epoch
        train(train_loader, model, optimizer, epoch)
        # evaluate on validation set
        acc = test(val_loader, model)

        # remember best acc and save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': max(acc, best_acc),
        }, acc > best_acc, save_directory)

        # Although the best saved model always updates no matter the quantity
        # of improvement, let's only count it if there was a big enough gain.
        if (acc - FLAGS.minimum_gain) > best_acc:
            best_epoch = epoch
            best_acc = acc

        epoch += 1

        # update learning rate
        scheduler.step()

    resume_filename = os.path.join(save_directory, 'model_best.pth.tar')
    _, best_val = load_checkpoint(model, resume_filename)
    acc = test(test_loader, model)
    print('Final acc - Test: {:.1f} Val: {:.1f}'.format(acc, best_val))

def train(train_loader, model, optimizer, epoch):
    average_loss = RunningAverage()
    steps_per_epoch = len(train_loader.dataset) // FLAGS.batch_size
    display_interval = int(np.floor(steps_per_epoch * FLAGS.display_interval))

    model.train()
    for batch_idx, (im_feats, sent_feats) in enumerate(train_loader):
        labels = np.repeat(np.eye(im_feats.size(0), dtype=np.float32), FLAGS.sample_size, axis=0)
        labels = torch.from_numpy(labels)
        if FLAGS.cuda:
            im_feats, sent_feats, labels = im_feats.cuda(), sent_feats.cuda(), labels.cuda()

        im_feats, sent_feats, labels = Variable(im_feats), Variable(sent_feats), Variable(labels)
        sent_feats = sent_feats.view(labels.size(0), -1)
        loss, i_embed, s_embed = model.train_forward(im_feats, sent_feats, labels)

        average_loss.update(loss.data.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % display_interval == 0:
            print('Epoch: {:d} Step: [{:d}/{:d}] Loss: {:f}'.format(epoch, batch_idx, steps_per_epoch, average_loss.avg()))

def test(test_loader, model):
    model.eval()
    sent_feats = torch.from_numpy(test_loader.sent_feats)
    im_feats = torch.from_numpy(test_loader.im_feats)
    if FLAGS.cuda:
        sent_feats, im_feats = sent_feats.cuda(), im_feats.cuda()

    sent_feats, im_feats = Variable(sent_feats), Variable(im_feats)
    images, sentences = model(im_feats, sent_feats)
    sentences, images = sentences.data, images.data
    im_labels = torch.from_numpy(test_loader.labels)

    dist_matrix = pdist(sentences, images)
    sent2im = recallAtK(dist_matrix, im_labels)
    im2sent = recallAtK(dist_matrix.t(), im_labels.t())
    recalls = np.append(im2sent, sent2im)
    mR = np.mean(recalls)
    print('\n{} set - mR: {:.1f} im2sent: {:.1f} {:.1f} {:.1f} sent2im: {:.1f} {:.1f} {:.1f}\n'.format(test_loader.split, mR, *recalls))
    return mR

def recallAtK(dist_matrix, labels):
    assert len(dist_matrix) == len(labels)
    thresholds = [1, 5, 10]
    successAtK = np.zeros(len(thresholds), np.float32)
    _, indices = dist_matrix.topk(max(thresholds), largest=False)
    for i, k in enumerate(thresholds):
        for sample_indices, sample_labels in zip(indices[:, :k], labels):
            successAtK[i] += sample_labels[sample_indices].max()

    if len(indices) > 0:
        successAtK /= len(indices)

    successAtK = np.round(successAtK*100, 1)
    return successAtK

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0.

    def update(self, val):
        self.value_sum += val
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average

def load_checkpoint(image_sentence_model, resume_filename):
    start_epoch = 1
    best_acc = 0.0
    if resume_filename:
        if os.path.isfile(resume_filename):
            print("=> loading checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            image_sentence_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_filename))

    return start_epoch, best_acc

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--name', type=str, default='Two_Branch_Network', help='Name of experiment')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--feat_path', type=str, default='data', help='Path to the cached features.')
    parser.add_argument('--dataset', type=str, default='flickr', help='Dataset we are training on.')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory for saving checkpoints.')
    parser.add_argument('--resume', type=str, help='Full path location of file to restore and resume training from')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='To only run inference on test set')
    # Training parameters.
    parser.add_argument('--language_model', type=str, default='avg', help='Type of language model to use. Supported: avg, attend')
    parser.add_argument('--display_interval', type=int, default=0.25, help='Portion of iterations before displaying loss.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--text_lr_multi', type=float, default=2.0, metavar='M',
                        help='learning rate multiplier for the text branch (default: 2.0)')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
    parser.add_argument('--sample_size', type=int, default=2, help='Number of positive pair to sample.')
    parser.add_argument('--max_num_epoch', type=int, default=-1, help='Max number of epochs to train, number < 0 means use automatic stopping criteria.')
    parser.add_argument('--no_gain_stop', type=int, default=10, metavar='N',
                        help='number of epochs used to perform early stopping based on validation performance (default: 10)')
    parser.add_argument('--minimum_gain', type=float, default=0.1, metavar='N',
                        help='minimum performance gain for a model to be considered better (default: 0.1)')
    parser.add_argument('--num_neg_sample', type=int, default=10, help='Number of negative example to sample.')
    parser.add_argument('--dim_embed', type=int, default=2048, metavar='N',
                        help='how many dimensions in embedding (default: 2048)')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin.')
    parser.add_argument('--word_embedding_reg', type=float, default=5e-5, help='Weight on the L2 regularization of the pretrained word embedding.')
    parser.add_argument('--im_loss_factor', type=float, default=1.5,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--sent_only_loss_factor', type=float, default=0.1,
                        help='Factor multiplied with sent only loss. Set to 0 for no neighbor constraint.')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    
    assert FLAGS.language_model in ['avg', 'attend']
    assert FLAGS.dataset in ['flickr', 'coco']
    FLAGS.cuda = not FLAGS.no_cuda and torch.cuda.is_available()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)

    main()

