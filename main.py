from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np

from data_loader import ImageSentenceFeatures
from model import ImageSentenceEmbeddingNetwork

# Training settings
parser = argparse.ArgumentParser(description='Two Branch Image Sentence Model')
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.05, metavar='M',
                    help='margin for triplet loss (default: 0.05)')
parser.add_argument('--sent_only_loss', type=float, default=0.05, metavar='M',
                    help='contribution of the sentence-to-sentence loss (default: 0.05)')
parser.add_argument('--diff_im_loss', type=float, default=1.5, metavar='M',
                    help='contribution to the loss for sentence-to-image retreival (default: 1.5)')
parser.add_argument('--num_negatives_to_sample', type=int, default=10, metavar='N',
                    help='number of negatives to subsample for the loss (default: 10)')
parser.add_argument('--text_lr_multi', type=float, default=2.0, metavar='M',
                    help='learning rate multiplier for the text branch (default: 2.0)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Two_Branch_Image_Sentence', type=str,
                    help='name of experiment')
parser.add_argument('--dim_embed', type=int, default=2048, metavar='N',
                    help='how many dimensions in embedding (default: 2048)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--resnet', dest='resnet', action='store_true',
                    help='Indicator of using resnet rather than vgg features')
parser.add_argument('--no_gain_stop', type=int, default=10, metavar='N',
                    help='number of epochs used to perform early stopping based on validation performance (default: 10)')
parser.add_argument('--minimum_gain', type=float, default=5e-1, metavar='N',
                    help='minimum performance gain for a model to be considered better (default: 5e-1)')
parser.add_argument('--max_epoch', type=int, default=-1, metavar='N',
                    help='maximum number of epochs, -1 indicates no limit (default: -1)')
parser.set_defaults(resnet=False)
parser.set_defaults(test=False)

def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.visdom:
        global plotter 
        plotter = VisdomLinePlotter(env_name=args.name)
    
    image_sentence_model = ImageSentenceEmbeddingNetwork(args);

    # optionally resume from a checkpoint
    start_epoch, best_acc = load_checkpoint(image_sentence_model, args.resume)
    cudnn.benchmark = True

    parameters = [{'params' : image_sentence_model.image_branch.fc1.parameters()},
                  {'params' : image_sentence_model.image_branch.fc2.parameters()},
                  {'params' : image_sentence_model.text_branch.fc1.parameters(), 'lr' : args.lr*args.text_lr_multi},
                  {'params' : image_sentence_model.text_branch.fc2.parameters(), 'lr' : args.lr*args.text_lr_multi}] 
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=0.001)
    n_parameters = sum([p.data.nelement() for p in image_sentence_model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    # Code assumes test data is not shuffled
    test_loader = torch.utils.data.DataLoader(ImageSentenceFeatures(args, 'test'),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    if args.test:
        test_acc = test(test_loader, image_sentence_model)
        sys.exit()

    train_loader = torch.utils.data.DataLoader(ImageSentenceFeatures(args, 'train'),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(ImageSentenceFeatures(args, 'val'),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    best_epoch = start_epoch
    epoch = start_epoch
    while (epoch - best_epoch) < args.no_gain_stop and (args.max_epoch < 0 or epoch <= args.max_epoch):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, image_sentence_model, optimizer, epoch)
        # evaluate on validation set
        acc = test(val_loader, image_sentence_model)

        # remember best acc and save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': image_sentence_model.state_dict(),
            'best_acc': max(acc, best_acc),
        }, acc > best_acc)

        # Although the best saved model always updates no matter the quantity
        # of improvement, let's only count it if there was a big enough gain.
        if (acc - args.minimum_gain) > best_acc:
            best_epoch = epoch
            best_acc = acc

        epoch += 1

    resume_filename = 'runs/%s/'%(args.name) + 'model_best.pth.tar'
    _, best_val = load_checkpoint(image_sentence_model, resume_filename)
    acc = test(test_loader, image_sentence_model)
    print('Final acc - Test: {:.1f} Val: {:.1f}'.format(acc, best_val))

def train(train_loader, image_sentence_model, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    image_sentence_model.train()
    for batch_idx, (img, sentence, im_label) in enumerate(train_loader):
        if args.cuda:
            img, sentence, im_label = img.cuda(), sentence.cuda(), im_label.cuda()

        im_label = im_label.expand(im_label.size(0),im_label.size(0))
        im_label = im_label == im_label.t()
        img, sentence, im_label = Variable(img), Variable(sentence), Variable(im_label)

        # Two sentences are returned for each image during training
        sentence = torch.cat([sentence[:,:6000],sentence[:,6000:]], dim=0)
        s_embed, i_embed = image_sentence_model(sentence, img)
        loss = 0
        n_images = i_embed.size(0)
        for index, row in enumerate(s_embed):
            im_id = index % n_images
            neg_pair = im_label[im_id,:] == 0
            dist_ims = torch.nn.functional.pairwise_distance(row.expand_as(i_embed), i_embed, 2)
            neg_ims = dist_ims[neg_pair]
            loss_diff_im = torch.clamp(args.margin + (dist_ims[im_id].expand_as(neg_ims) - neg_ims), min=0, max=1e6)
            loss_diff_im, _ = loss_diff_im.topk(args.num_negatives_to_sample)
            loss_diff_im = loss_diff_im.mean()

            neg_pair = torch.cat([neg_pair, neg_pair])
            dist_sent = torch.nn.functional.pairwise_distance(i_embed[im_id,:].expand_as(s_embed), s_embed, 2)
            neg_sent = dist_sent[neg_pair]
            loss_diff_sent = torch.clamp(args.margin + (dist_sent[index].expand_as(neg_sent) - neg_sent), min=0, max=1e6)
            loss_diff_sent, _ = loss_diff_sent.topk(args.num_negatives_to_sample)
            loss_diff_sent = loss_diff_sent.mean()

            dist_sent_only = torch.nn.functional.pairwise_distance(row.expand_as(s_embed), s_embed, 2)
            pos_pair = im_label[im_id,:]
            pos_pair = torch.cat([pos_pair, pos_pair])
            pos_pair = torch.max(dist_sent_only[pos_pair])
            neg_sent = dist_sent_only[neg_pair]
            loss_sent_only = torch.clamp(args.margin + (pos_pair.expand_as(neg_sent) - neg_sent), min=0, max=1e6)
            loss_sent_only, _ = loss_sent_only.topk(args.num_negatives_to_sample)
            loss_sent_only = loss_sent_only.mean()
            total_loss = (loss_diff_sent + loss_diff_im * args.diff_im_loss + loss_sent_only * args.sent_only_loss)
            loss += total_loss

        loss /= s_embed.size(0)
        losses.update(loss.data[0], s_embed.size(0)*args.num_negatives_to_sample)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f})'.format(
                      epoch, batch_idx * i_embed.size(0), len(train_loader.dataset),
                      losses.val, losses.avg))

    # log avg values to visdom
    if args.visdom:
        plotter.plot('loss', 'train', epoch, losses.avg)

def test(test_loader, image_sentence_model):
    # switch to evaluation mode
    image_sentence_model.eval()

    # get embedded features for the entire split
    images = []
    sentences = []
    im_labels = []
    for batch_idx, (img, sentence, im_label) in enumerate(test_loader):
        im_labels.append(torch.cat([im_label, im_label, im_label, im_label, im_label]))
        if args.cuda:
            img, sentence = img.cuda(), sentence.cuda()
        img, sentence = Variable(img), Variable(sentence)
        
        # All five sentences for an image are returned on the test/val sets
        sentence = torch.cat([sentence[:,:6000],sentence[:,6000:12000],sentence[:,12000:18000],sentence[:,18000:24000],sentence[:,24000:]], dim=0)
        s_embed, i_embed = image_sentence_model(sentence, img)
        sentences.append(s_embed)
        images.append(i_embed)

    # eval performance
    im_labels = torch.cat(im_labels).cuda()
    sentences = torch.cat(sentences)
    images = torch.cat(images)
    dist_matrix = []
    recallAt = [1,5,10]
    max_topk = max(recallAt)
    res_im2sent = np.zeros(len(recallAt))
    for index, example in enumerate(images):
        dist = torch.nn.functional.pairwise_distance(example.expand_as(sentences), sentences, 2)
        _, pred = dist.topk(max_topk, 0, largest=False, sorted=True)
        pred = im_labels[torch.squeeze(pred.data)]
        correct = pred == index
        for k_id, k in enumerate(recallAt):
            res_im2sent[k_id] += max(correct[:k].view(-1).float())
            
        dist_matrix.append(dist.data)

    res_im2sent = np.round((res_im2sent/images.size(0))*100.0, 1)
    dist_matrix = torch.squeeze(torch.stack(dist_matrix))
    res_sent2im = np.zeros(len(recallAt))
    for index in range(sentences.size(0)):
        _, pred = dist_matrix[:,index].topk(max_topk, 0, largest=False, sorted=True)
        target = im_labels[index]
        correct = pred == target
        for k_id, k in enumerate(recallAt):
            res_sent2im[k_id] += max(correct[:k].view(-1).float())

    res_sent2im = np.round((res_sent2im/sentences.size(0))*100.0, 1)

    acc = np.sum(res_im2sent + res_sent2im)
    print('\n{} set - Total: {:.1f} im2sent: {:.1f} {:.1f} {:.1f} sent2im: {:.1f} {:.1f} {:.1f}\n'.format(test_loader.dataset.split, acc, res_im2sent[0], res_im2sent[1], res_im2sent[2], res_sent2im[0], res_sent2im[1], res_sent2im[2]))
    return acc

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
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 1
    adjustment_factor = int(np.floor(epoch / 10.0))
    for i in range(adjustment_factor):
        lr *= .1

    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr*args.lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr
        
if __name__ == '__main__':
    main()    
