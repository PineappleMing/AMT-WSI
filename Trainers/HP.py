import numpy as np


def loss_schedule(max_epoch):
    start_2 = int(0.2 * max_epoch)
    start_3 = int(0.4 * max_epoch)
    duration = int(0.5 * max_epoch)

    weight_1 = [[[1.0, 1.0, (epoch - start_2) / duration, (epoch - start_2) / duration]] for epoch in range(max_epoch)]
    weight_2 = [[[epoch >= start_2, epoch >= start_2, (epoch - start_3) / duration, (epoch - start_3) / duration]] for
                epoch in range(max_epoch)]
    weight_3 = [[[epoch >= start_3, epoch >= start_3, 0.0, 0.0]] for epoch in range(max_epoch)]
    weight = np.hstack((weight_1, weight_2, weight_3))
    weight = np.clip(weight, 0.0, 1.0)
    return weight


models_path = "/nfs3-p1/lhm/model_save/"
log_dir = "/nfs3-p1/lhm/log/"

temp = 5
