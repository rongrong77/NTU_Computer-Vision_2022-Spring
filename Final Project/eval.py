import os, sys, argparse
import numpy as np


def calculate_dist(label, pred):
    assert label.shape[0] == pred.shape[0], 'The number of predicted results should be the same as the number of ground truth.'
    dist = np.sqrt(np.sum((label-pred)**2, axis=1))
    dist = np.mean(dist)
    return dist

def benchmark(gt_path, test_path, mode='train'):
    if mode == 'train':
        for seq in ['seq1', 'seq2', 'seq3']:
            label = np.loadtxt(os.path.join(gt_path, seq, 'gt_pose.txt'))
            pred = np.loadtxt(os.path.join(test_path, f'{seq}/pred_pose.txt'))
            score = calculate_dist(label, pred)
            print(f'{seq} mean Error of {seq}: {score:.5f}')
    if mode == 'test':
        for seq in ['test1', 'test2']:
            label = np.loadtxt(os.path.join(gt_path, seq, 'gt_pose.txt'))
            pred = np.loadtxt(os.path.join(test_path, f'{seq}/pred_pose.txt'))
            score = calculate_dist(label, pred)
            print(f'{seq} mean Error of {seq}: {score:.5f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--result_name', type=str, default='TEST_ST_12')
    opt = parser.parse_args()
    print(opt)

    gt_path = './ITRI_DLC2/'
    test_path = f'./result/{opt.result_name}/solution/'

    benchmark(gt_path, test_path, mode=opt.mode)