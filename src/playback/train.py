import torch
from dataset import PlaybackDataset
import metrics
import model
import numpy as np
from tqdm.auto import tqdm
import argparse
import helper
import os
import datetime
def inference(dataset, BATCH_SIZE, model):

    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    metric_calculator = metrics.AudioMetrics(rate=44100)
    Metric = []
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            metric = helper.test(model, sample, device)
            Metric.append(metric)
    avg_metric = np.round(np.mean(np.concatenate(Metric, axis=0), axis=0),2).tolist()
    print(avg_metric)
    return avg_metric

def train(dataset, EPOCH, lr, BATCH_SIZE, model,):
    train_dataset, test_dataset = dataset

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True,
                                               drop_last=True, pin_memory=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    save_dir = 'checkpoints/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(save_dir)
    loss_best = 1000
    ckpt_best = model.state_dict()
    if checkpoint is not None:
        print('first test the initial checkpoint')
        avg_metric = inference(test_dataset, BATCH_SIZE, model)
    for e in range(EPOCH):
        Loss_list = []
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for i, sample in enumerate(train_loader):
                loss = helper.train(model, sample, optimizer, device)
                Loss_list.append(loss)
                t.set_description('Epoch %i' % e)
                t.set_postfix(loss=np.mean(Loss_list))
                t.update(1)
        mean_lost = np.mean(Loss_list)
        avg_metric = inference(test_dataset, BATCH_SIZE, model)
        if mean_lost < loss_best:
            ckpt_best = model.state_dict()
            loss_best = mean_lost
            metric_best = avg_metric
            torch.save(ckpt_best, save_dir + args.model + '_' + str(e) + '_' + str(metric_best) + '.pth')
    torch.save(ckpt_best, save_dir + 'best.pth')
    print('best performance is', metric_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", default=False, required=False)
    parser.add_argument('--model', action="store", type=str, default='SuDORMRF', required=False,
     help='choose the model', choices=['SuDORMRF', 'TasNet', 'Sepformer'])

    args = parser.parse_args()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = getattr(model, args.model)().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    BATCH_SIZE = 8
    lr = 0.0001
    EPOCH = 50
    checkpoint = None

    dataset = PlaybackDataset()
    print('dataset length is', len(dataset))
    if checkpoint is not None:
        ckpt = torch.load('checkpoints/' + checkpoint)
        model.load_state_dict(ckpt, strict=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    if args.train:
        train([train_dataset, test_dataset], EPOCH, lr, BATCH_SIZE, model)
    else:
        inference(test_dataset, BATCH_SIZE, model)

      