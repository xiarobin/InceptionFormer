import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.train_utils import *
from dataset import TreeCompletion3D
import h5py

from tqdm import tqdm

def save_h5(data, path):
    f = h5py.File(path, 'w')
    a = data.data.cpu().numpy()
    # print(a.shape)
    f.create_dataset('data', data=a)
    f.close()

    with h5py.File(path, 'r') as f:
        dataset = f['data']
        data = dataset[:]

    txt_file_path = os.path.splitext(path)[0] + '.txt'

    with open(txt_file_path, 'w') as txt_file:
        if len(data.shape) > 1:
            for row in data:
                txt_file.write(' '.join(map(str, row)) + '\n')
        else:
            for value in data:
                txt_file.write(f'{value}\n')

def save_obj(point, path):
    n = point.shape[0]
    with open(path, 'w') as f:
        for i in range(n):
            f.write("v {0} {1} {2}\n".format(point[i][0],point[i][1],point[i][2]))
    f.close()

def test():
    dataset_test = TreeCompletion3D(args.TreeCompletion3Dpath, prefix="Test")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=1, drop_last=True)
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_test, desc="Testing Progress"), 0):

            label, inputs_cpu, gt_cpu, pathname = data
            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, is_training=False)

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                if not os.path.isdir(os.path.join(os.path.dirname(args.load_model), '4096')):
                    os.makedirs(os.path.join(os.path.dirname(args.load_model), '4096'))

                for j in range(args.batch_size):
                    path = os.path.join(os.path.dirname(args.load_model), '4096', str(pathname[j]))
                    save_h5(result_dict['out2'][j], path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs',arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
