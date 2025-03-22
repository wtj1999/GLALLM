from data_provider.data_loader import *#Dataset_Custom, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import DataLoader

data_dict = {
    'dswe1': Dataset_DSWE1,
    'pems04': Dataset_PEMS,
    'dswe2': Dataset_DSWE1,
    'nrel': Dataset_DSWE1,
    'sdwpf': Dataset_DSWE1,
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    Data = data_dict[args.dataset]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = True
        drop_last = drop_last_test
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        percent=percent
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
