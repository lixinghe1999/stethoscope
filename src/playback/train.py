import torch
from dataset import PublicDataset, PairedDataset
# import metrics
import model
import numpy as np
import argparse
import os
import datetime
def inference(dataset, BATCH_SIZE, model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    Metric = []
    model.eval()
    with torch.no_grad():
        Metric = model.run_epoch_test(test_loader, device)
    avg_metric = np.round(np.mean(np.array(Metric), axis=0),3).tolist()
    print(avg_metric)
    return avg_metric

def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    [large_dataset, train_dataset], test_dataset = dataset

    large_loader = torch.utils.data.DataLoader(dataset=large_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=False)
    small_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=False)
    save_dir = 'checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(save_dir)
    loss_best = 1000
    ckpt_best = model.state_dict()
    if checkpoint is not None:
        print('first test the initial checkpoint')
        avg_metric = inference(test_dataset, BATCH_SIZE, model)
    for e in range(EPOCH):
        Loss_list = model.run_epoch_train(small_loader, large_loader, device)
        mean_lost = np.mean(Loss_list)
        avg_metric = inference(test_dataset, BATCH_SIZE, model)
        if e % 10 == 0:
            torch.save(model.state_dict(), save_dir + args.task + '_' + args.model + '_' + str(e) + '_' + str(avg_metric) + '.pth')
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
    torch.save(ckpt_best, save_dir + 'best.pth')
    print('best performance is', metric_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--task', action="store", type=str, default='Reverse', required=False)
    parser.add_argument('--model', action="store", type=str, default='SuDORMRF', required=False,
     help='choose the model', choices=['SuDORMRF', 'TasNet', 'Sepformer'])

    args = parser.parse_args()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    model = getattr(model, args.task)(getattr(model, args.model)).to(device)
    BATCH_SIZE = 8
    lr = 0.0001
    EPOCH = 50
    checkpoint = None

    large_dataset = PublicDataset()
    small_dataset = PairedDataset()
    if checkpoint is not None:
        ckpt = torch.load('checkpoints/' + checkpoint)
        model.load_state_dict(ckpt, strict=True)
    train_dataset, test_dataset = torch.utils.data.random_split(small_dataset, [int(len(small_dataset) * 0.8), 
                                                                                len(small_dataset) - int(len(small_dataset) * 0.8)])
    if args.train:
        train([[large_dataset, train_dataset], test_dataset], EPOCH, lr, BATCH_SIZE, model)
    else:
        inference(test_dataset, BATCH_SIZE, model)

      