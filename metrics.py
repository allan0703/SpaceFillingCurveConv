from __future__ import division

import torch

__all__ = ['Accuracy', 'ClassAccuracy', 'LossMean']


class ConfusionMatrix(object):
    """
    Calculates the confusion matrix of a multiclass classifier
    The class tracks 2 objects:
        1- cm: Confusion matrix given data
        2- total_seen: Total number of examples seen

    The 3 major functions to use are:
        1- update: Takes as input the logits and labels and counts
                   predictions in the confusion matrix
        2- result: Returns value based on metric given (accuracy, class_accuracy, mIoU)
        3- reset: Zeros the values of cm and total_seen
    """
    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        # create class variables:
        self.num_classes = num_classes
        self._cm = None
        self._total_seen = None

        # we must reset values at initialization:
        self.reset()

    def reset(self):
        self._cm = torch.zeros(self.num_classes, self.num_classes,
                              dtype=torch.int64, device='cpu')
        self._total_seen = 0

    def update(self,
               y_true: torch.Tensor,
               y_logits: torch.Tensor):
        """
        Update the correct and total values seen

        :param y_true: torch tensor with true labels (batch_size x ... x 1)
        :param y_logits: torch tensor with output logits (batch_size x ... x C)
        """
        # check first we got the right sizes:
        if not (y_logits.ndimension() == y_true.ndimension() + 1):
            raise RuntimeError('y_true should be of shape (batch_size,) and y_logits '
                               'of shape (batch_size, C) but given shapes are {} and {}'
                               .format(y_true.size(), y_logits.size()))

        # get predictions from logits and compute correct
        # print(y_logits.size(), y_true.size())
        preds = torch.argmax(y_logits, dim=1).flatten()
        labels = y_true.flatten()
        # print(preds.size(), labels.size())

        # make sure we don't have wrong labels
        mask = (labels >= 0) & (labels < self.num_classes)
        labels = labels[mask]
        preds = preds[mask]

        indices = self.num_classes * labels + preds
        cm = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

        self._cm += cm.to(self._cm)
        self._total_seen += y_logits.size(0)

    def result(self, metric='accuracy', part_category=None):
        # make sure we have seen examples
        if self._total_seen == 0:
            raise RuntimeError('Object must have at least one example before computing accuracy')

        self._cm = self._cm.float()
        if metric == 'accuracy':
            return self._cm.diag().sum() / (self._cm.sum() + 1e-15)
        if metric == 'class_accuracy':
            return self._cm.diag() / (self._cm.sum(dim=1) + 1e-15)
        if metric == 'iou':
            return self._cm.diag() / (self._cm.sum(dim=1) + self._cm.sum(dim=0) - self._cm.diag() + 1e-15)
        # if metric == 'iou_category':
        #     all_categories = torch.unique(part_category[:, 1])
        #     for category in all_categories
        #         part_idx = torch.where(part_category[:, 1] == category)
        #         total_correct = 0.0
        #         total_seen = 0
        #         total_pred = 0
        #         for idx in part_idx:
        #             total_correct += self._cm[idx, idx]
        return ValueError('metric should be accuracy, class_accuracy, or iou, but {} was give'.format(metric))


class Accuracy(object):
    """
    Calculates the accuracy of a multiclass classifier
    The class tracks 2 objects:
        1- total_correct: Number of correct classified examples
        2- total_seen: Total number of examples seen

    The 3 major functions to use are:
        1- update: Takes as input the logits and labels, computes
                   the number of correct predictions, updates
                   total_correct and total_seen
        2- result: Returns current accuracy (total_correct / total_seen)
        3- reset: Zeros the values of total_correct and total_seen
    """
    def __init__(self):
        super(Accuracy, self).__init__()
        # create class variables:
        self._total_correct = None
        self._total_seen = None

        # we must reset values at initialization:
        self.reset()

    def reset(self):
        self._total_correct = 0
        self._total_seen = 0

    def update(self,
               y_true: torch.Tensor,
               y_logits: torch.Tensor):
        """
        Update the correct and total values seen

        :param y_true: torch tensor with true labels (batch_size x ... x 1)
        :param y_logits: torch tensor with output logits (batch_size x ... x C)
        """
        # check first we got the right sizes:
        if not (y_logits.ndimension() == y_true.ndimension() + 1):
            raise RuntimeError('y_true should be of shape (batch_size,) and y_logits '
                               'of shape (batch_size, C) but given shapes are {} and {}'
                               .format(y_true.size(), y_logits.size()))

        # get predictions from logits and compute correct
        preds = torch.argmax(y_logits, dim=-1)
        correct = torch.eq(preds, y_true).view(-1)

        # update total correct and seen:
        self._total_correct += torch.sum(correct).item()
        self._total_seen += correct.size(0)

    def result(self):
        # make sure we have seen examples
        if self._total_seen == 0:
            raise RuntimeError('Object must have at least one example before computing accuracy')
        return self._total_correct / self._total_seen


class ClassAccuracy(object):
    """
    Calculates the per-class accuracy of a multiclass classifier
    The class tracks 2 objects:
        1- total_correct: Number of correct classified examples per class
        2- total_seen: Total number of examples seen per class

    The 3 major functions to use are:
        1- update: Takes as input the logits and labels, computes
                   the number of correct predictions per class,
                   updates total_correct and total_seen
        2- result: Returns current accuracy sum(total_correct / total_seen)
        3- reset: Zeros the values of total_correct and total_seen
    """
    def __init__(self, num_classes):
        super(ClassAccuracy, self).__init__()
        # create class variables:
        self._total_correct = None
        self._total_seen = None
        self._num_classes = num_classes

        # we must reset values at initialization:
        self.reset()

    def reset(self):
        self._total_correct = torch.zeros((self._num_classes, 1))
        self._total_seen = torch.zeros((self._num_classes, 1))

    def update(self,
               y_true: torch.Tensor,
               y_logits: torch.Tensor):
        """
        Update the correct and total values seen

        :param y_true: torch tensor with true labels (batch_sizex1)
        :param y_logits: torch tensor with output logits (batch_sizexC)
        """
        # check first we got the right sizes:
        if not (y_logits.ndimension() == y_true.ndimension() + 1 and y_logits.ndimension() == 2):
            raise RuntimeError('y_true should be of shape (batch_size,) and y_logits '
                               'of shape (batch_size, C) but given shapes are {} and {}'
                               .format(y_true.size(), y_logits.size()))

        # get predictions from logits and compute correct
        preds = torch.argmax(y_logits, dim=-1)
        for i in range(self._num_classes):
            idx = torch.eq(y_true, i).view(-1)
            correct = torch.eq(preds[idx, ...], y_true[idx, ...]).view(-1)
            self._total_correct[i] += torch.sum(correct).item()
            self._total_seen[i] += correct.size(0)

    def result(self):
        # make sure we have seen examples
        if torch.sum(self._total_seen) == 0:
            raise RuntimeError('Object must have at least one example before computing accuracy')
        return torch.mean(self._total_correct / self._total_seen).item()


class LossMean(object):
    def __init__(self):
        super(LossMean, self).__init__()
        self._sum_loss = None
        self._total_seen = None
        self.reset()

    def reset(self):
        self._sum_loss = 0.0
        self._total_seen = 0

    def update(self, average_loss: torch.Tensor, batch_size: int):
        # update loss sum using average loss provided * batch_size
        self._sum_loss += average_loss.item() * batch_size
        self._total_seen += batch_size

    def result(self):
        # make sure we have seen examples
        if self._total_seen == 0:
            raise RuntimeError('Object must have at least one example before computing average loss')
        return self._sum_loss / self._total_seen


if __name__ == '__main__':
    accuracy = Accuracy()
    batch_size = 10
    num_classes = 3
    num_points = 50
    y_true = torch.randint(0, num_classes, (batch_size, num_points), dtype=torch.long)
    y_pred = torch.rand((batch_size, num_classes, num_points), dtype=torch.float)

    # accuracy.update(y_true, y_pred)
    #
    # res = accuracy.result()
    #
    # print('Accuracy: {}'.format(res))

    cm = ConfusionMatrix(num_classes=num_classes)
    cm.update(y_true, y_pred)

    res = cm.result(metric='class_accuracy')
    print('CM Accuracy: {}'.format(res.mean()))

    # class_accuracy = ClassAccuracy(num_classes=num_classes)
    # class_accuracy.update(y_true, y_pred)
    #
    # res = class_accuracy.result()
    #
    # print('Class Accuracy: {}'.format(res))
    #
    # loss = LossMean()
    # average_loss = torch.rand((1, ), dtype=torch.float)
    #
    # loss.update(average_loss, batch_size)
    #
    # res = loss.result()
    # print('Loss: {}'.format(res))
