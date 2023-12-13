import torch
from dataset import PublicDataset, PairedDataset
# import metrics
import model
import numpy as np
import argparse
import os
import datetime
def fine_grained_inference(model):
    people = ['Lixing_He',]
    smartphone = ['PixelXL', 'Pixel6', 'iPhone13', 'EDIFIER', 'SAST']
    textile = ['skin', 'cotton', 'polyester', 'thickcotton', 'thickpolyester', 'PU', 'cowboy']
    for p in people:
        test_dataset = PairedDataset(people=[p], train=False)
        avg_metric = inference(test_dataset, model)
        print(p, avg_metric)
    for s in smartphone:
        test_dataset = PairedDataset(phone=[s], train=False)
        avg_metric = inference(test_dataset, model)
        print(s, avg_metric)
    for t in textile:
        test_dataset = PairedDataset(textile=[t], train=False)
        avg_metric = inference(test_dataset, model)
        print(t, avg_metric)
                
def inference(dataset, model):
    test_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=4, batch_size=4, shuffle=False, drop_last=False)
    Metric = []
    model.eval()
    with torch.no_grad():
        Metric = model.run_epoch_test(test_loader, device)
    avg_metric = np.round(np.mean(np.array(Metric), axis=0),3).tolist()
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
        avg_metric = inference(test_dataset, model)
        print(avg_metric)

    for e in range(EPOCH):
        Loss_list = model.run_epoch_train(small_loader, large_loader, device)
        mean_lost = np.mean(Loss_list)
        avg_metric = inference(test_dataset, model)
        print(avg_metric)

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
    EPOCH = 20

    large_dataset = PublicDataset()
    train_dataset, test_dataset = PairedDataset(train=True), PairedDataset(train=False)
    print(len(train_dataset), len(test_dataset))
    checkpoint = None
    # checkpoint = '20231211-034927'
    if checkpoint is not None:
        ckpt = torch.load('checkpoints/' + checkpoint + '/best.pth')
        model.load_state_dict(ckpt, strict=True)
    if args.train:
        train([[large_dataset, train_dataset], test_dataset], EPOCH, lr, BATCH_SIZE, model)
    else:
        arg_metric = inference(test_dataset, model)
        print(arg_metric)
    fine_grained_inference(model)

      