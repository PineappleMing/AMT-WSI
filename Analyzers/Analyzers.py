import torch


class CommonAnalyzer:
    __slots__ = ['_writer', '_label_count', '_pred_count', '_patch_hit', '_patch_total', 'acc', 'label_rate',
                 'patch_acc']

    def __init__(self, writer):
        super().__init__()
        self._writer = writer
        self._label_count = [torch.tensor([]).long()] * 3
        self._pred_count = [torch.tensor([]).long()] * 3
        self._patch_hit = [0.] * 3
        self._patch_total = [0.] * 3
        self.acc = [[0., 0.]] * 3
        self.label_rate = [0.] * 3
        self.patch_acc = [0.] * 3

    def updateStageOne(self, label, pred, patch_label, fine_index):

        self._label_count[0] = torch.cat((self._label_count[0], label.flatten().cpu()))
        self._pred_count[0] = torch.cat(
            (self._pred_count[0], pred.argmax(dim=1).cpu()))  # 预测值[cls为0的得分，cls为1的得分]取argmax得到预测值
        self.acc[0] = [
            (((self._pred_count[0] == self._label_count[0]) & (self._label_count[0] == cls)).float().sum() / (
                    self._label_count[0] == cls).float().sum()).item() // 0.0001 / 100 for cls in
            range(2)]
        self.label_rate[0] = self._label_count[0].float().mean().item()
        B, num_focus = fine_index.shape
        for b in range(B):
            self._patch_hit[0] += (patch_label[b][fine_index[b]] == 1).sum()
            self._patch_total[0] += num_focus
        self.patch_acc[0] = self._patch_hit[0] / (self._patch_total[0] + 0.01)

    def updateStageTwo(self, label, pred, patch_label, fine_index):
        K, num_focus = fine_index.shape
        self._label_count[1] = torch.cat((self._label_count[1], label.flatten().cpu()))
        self._pred_count[1] = torch.cat(
            (self._pred_count[1], pred.argmax(dim=1).cpu()))  # 预测值[cls为0的得分，cls为1的得分]取argmax得到预测值
        self.acc[1] = [
            (((self._pred_count[1] == self._label_count[1]) & (self._label_count[1] == cls)).float().sum() / (
                    self._label_count[1] == cls).float().sum()).item() // 0.0001 / 100 for cls in
            range(2)]
        self.label_rate[1] = self._label_count[1].float().mean().item()
        for b in range(K):
            self._patch_hit[1] += (patch_label[b][fine_index[b]] == 1).sum()
            self._patch_total[1] += num_focus
        self.patch_acc[1] = self._patch_hit[1] / (self._patch_total[1] + 0.01)

    def updateStageThree(self, label, pred):
        K, _ = pred.shape
        self._label_count[2] = torch.cat((self._label_count[2], label.flatten().cpu()))
        self._pred_count[2] = torch.cat(
            (self._pred_count[2], pred.argmax(dim=1).cpu()))  # 预测值[cls为0的得分，cls为1的得分]取argmax得到预测值
        self.acc[2] = [
            (((self._pred_count[2] == self._label_count[2]) & (self._label_count[2] == cls)).float().sum() / (
                    self._label_count[2] == cls).float().sum()).item() // 0.0001 / 100 for cls in
            range(2)]
        self.label_rate[2] = self._label_count[2].float().mean().item()

    def saveToFile(self, *, step, mode='train'):
        self._writer.add_scalar(mode + '/stage1/acc_n', scalar_value=self.acc[0][0],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage1/acc_p', scalar_value=self.acc[0][1],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage1/patch_acc', scalar_value=self.patch_acc[0],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage2/acc_n', scalar_value=self.acc[1][0],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage2/acc_p', scalar_value=self.acc[1][1],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage2/patch_acc', scalar_value=self.patch_acc[1],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage3/acc_n', scalar_value=self.acc[2][0],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage3/acc_p', scalar_value=self.acc[2][1],
                                global_step=step)
        self._writer.add_scalar(mode + '/stage3/label_rate', scalar_value=self.label_rate[2],
                                global_step=step)

    def print(self, epoch, loss_se):
        print('\nEPOCH: {} STAGE 1: '.format(epoch))
        print(
            '\rlabel rate: {0:.4f}  pred rate: {1:.4f}; pos patch acc: {3:.4f}; acc: {2}'.format(
                self._label_count[0].float().mean().item(), self._pred_count[0].float().mean().item(),
                self.acc[0],
                self.patch_acc[0]), end='')
        if loss_se[1][0] == 0:
            return
        print('\nEPOCH: {} STAGE 2: '.format(epoch))
        print(
            '\rlabel rate: {0:.4f}  pred rate: {1:.4f}; pos patch acc: {3:.4f}; acc: {2}'.format(
                self._label_count[1].float().mean().item(), self._pred_count[1].float().mean().item(),
                self.acc[1],
                self.patch_acc[1]), end='')
        if loss_se[2][0] == 0:
            return
        print('\nEPOCH: {} STAGE 3: '.format(epoch))
        print(
            '\rlabel rate: {0:.4f}  pred rate: {1:.4f}; pos patch acc: {3:.4f}; acc: {2}'.format(
                self._label_count[2].float().mean().item(), self._pred_count[2].float().mean().item(),
                self.acc[2],
                self.patch_acc[2]), end='')
