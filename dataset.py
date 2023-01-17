class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root='', train=True, meta_eval=False, meta=True, num_meta=1000,
                 corruption_prob=0, corruption_type='sym', transform=None, target_transform=None,
                 download=False, seed=1):
        self.count = 0
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.meta = meta
        self.corruption_prob = corruption_prob
        self.num_meta = num_meta
        self.corruption_type = corruption_type
        self.meta_eval = meta_eval

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data_origin = []
            self.train_data = []
            self.train_labels = []
            self.train_labels_origin = []
            self.train_coarse_labels = []
            self.train_labels_true = []
            self.soft_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels_origin += entry['labels']
                    self.train_labels_true += entry['labels']
                    img_num_list = [int(self.num_meta / 10)] * 10
                    num_classes = 10
                else:
                    self.train_labels_origin += entry['fine_labels']
                    self.train_labels_true += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    img_num_list = [int(self.num_meta / 100)] * 100
                    num_classes = 100
                fo.close()

            self.train_data_origin = np.concatenate(self.train_data)
            self.train_data_origin = self.train_data_origin.reshape((50000, 3, 32, 32))
            self.train_data_origin = self.train_data_origin.transpose((0, 2, 3, 1))  # convert to HWC
            if self.corruption_type == 'flip':
                idx_to_meta = np.load('meta_list_flip' + str(self.corruption_prob) + '.npy')
                self.train_labels_origin = np.load('cifar10_flip_' + str(self.corruption_prob) + '.npy')
            if self.corruption_type == 'sym':
                idx_to_meta = np.load('meta_list_sym' + str(self.corruption_prob) + '.npy')
                self.train_labels_origin = np.load('cifar10_sym_' + str(self.corruption_prob) + '.npy')

            print(len(idx_to_meta))
            Noise_list = []
            for i in range(50000):
                Noise_list.append(i)
            idx_to_train = list(set(Noise_list).difference(set(idx_to_meta)))
            # clean_labels = self.train_labels[idx_to_train]
            # self.train_labels_true = list(np.array(self.train_labels_true)[idx_to_train])
            print("Print noisy label generation statistics:")
            print(self.corruption_type)
            for i in range(10):
                n_noisy = np.sum(np.array(self.train_labels_origin) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))

            if meta is True:
                self.train_data = self.train_data_origin[idx_to_meta]
                self.train_labels = list(np.array(self.train_labels_origin)[idx_to_meta])
            else:
                self.train_data = self.train_data_origin[idx_to_train]
                self.train_labels = list(np.array(self.train_labels_origin)[idx_to_train])
                self.soft_labels = list(np.zeros((len(self.train_data), num_classes), dtype=np.float32))
                self.prediction = np.zeros((len(self.train_data), 10, num_classes), dtype=np.float32)

        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def data_update(self, data_update_list):
        self.train_data = self.train_data_origin[data_update_list]
        self.train_labels = list(np.array(self.train_labels_origin)[data_update_list])

    def label_update(self, results, warmup):
        self.count += 1
        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10  # 10 #10
        self.prediction[:, idx] = results
        # self.prediction[:] =results
        # print(self.prediction)

        if self.count == warmup - 1:  # 79
            self.soft_labels = self.prediction.mean(axis=1)
            # print(self.soft_labels.shape)
            # print(self.soft_labels)
            # self.soft_labels = list(np.argmax(self.soft_labels, axis=1).astype(np.int64))
        if self.count > warmup - 1:
            self.soft_labels = results
            # self.soft_labels = list(np.argmax(self.soft_labels, axis=1).astype(np.int64))

    def __getitem__(self, index):
        if self.meta_eval:
            img, target = self.train_data_origin[index], self.train_labels_origin[index]
        else:
            if self.train:
                if self.meta:
                    # print(self.train_labels[index])
                    img, target, target_true = self.train_data[index], self.train_labels[index], self.train_labels_true[
                        index]
                else:
                    img, target, target_true = self.train_data[index], self.train_labels[index], self.train_labels_true[
                        index]
                    soft_labels = self.soft_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.meta_eval:
            return img, target
        else:
            if self.train:
                if self.meta:
                    return img, target
                else:
                    return img, target, target_true, soft_labels, index
            else:
                return img, target

    def __len__(self):
        if self.meta_eval:
            return 50000
        else:
            if self.train:
                if self.meta is True:
                    return self.num_meta
                else:
                    return 50000 - self.num_meta
            else:
                return 10000

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)