# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class OCC_REID(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir_train = 'Market-1501'
    dataset_dir_test = 'Occluded_REID'
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(OCC_REID, self).__init__()
        self.dataset_dir_train = osp.join(root, self.dataset_dir_train)
        self.dataset_dir_test = osp.join(root, self.dataset_dir_test)
        self.train_dir = osp.join(self.dataset_dir_train, 'bounding_box_train')
        
        self.query_dir = osp.join(self.dataset_dir_test, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir_test, 'whole_body_images')

        # self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False, is_query=True)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)
        # print(gallery)
        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            # img_path = '/media/userdisk1/jbl/after_0415/dataset/Occluded_REID/occluded_body_images/001/001_01.tif'
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset


    def _process_dir_test(self, dir_path, relabel=False, is_query=False):
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        # pid_container = set()
        dataset = []
        import os
        filelist = os.listdir(dir_path)
        for file in filelist:
            path_1 = os.path.join(dir_path, file)
            filelist_1 = os.listdir(path_1)
            for file_1 in filelist_1:
                img_path = os.path.join(path_1, file_1)
                pid = int(file)
                pid -= 1
                camid = 0 if is_query else 1
                dataset.append((img_path, pid, camid, 1))
        return dataset