import torch
from torch import nn
import numpy as np
from AVContrastLoss import AVContrastLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


class VectorsDataset(Dataset):
    def __init__(self, mode='image'):
        self.classes = {'alarm clock': 0, 'clock': 1, 'bee': 2, 'bird': 3, 'owl': 4,
                        'phone': 5, 'church': 6, 'cow': 7, 'duck': 8, 'dog': 9,
                        'frog': 10, 'horse': 11, 'keyboard': 12, 'pencil': 13, 'guitar': 14,
                        'piano': 15, 'train': 16, 'violin': 17, 'clarinet': 18, 'pig': 19}
        self.classes_list = ['alarm clock', 'clock', 'bee', 'bird', 'owl',
                        'phone', 'church', 'cow', 'duck', 'dog',
                        'frog', 'horse', 'keyboard', 'pencil', 'guitar',
                        'piano', 'train', 'violin', 'clarinet', 'pig']
        self.mode = mode
        self.features, self.labels = self.load_features(mode=mode)
        self.counter = 0

    def __getitem__(self, i):
        # Get i-th path.
        feature, label = self.features[i], self.labels[i]

        return feature, label

    def __len__(self):
        return len(self.labels)

    def load_features(self, mode):
        if mode == 'image':
            feature = np.load('image_classifier/v_features.npy')
            label = np.load('image_classifier/v_labels.npy')
        elif mode == 'audio':
            feature = np.load('sound_classifier/a_features.npy')
            label = np.load('sound_classifier/a_labels.npy')
        else:
            raise AssertionError
        return feature, label


class VectorMapper(nn.Module):
    def __init__(self, vector_dim=128, mode={'img2snd': 0, 'snd2img': 1}, pretrained=False):
        super(VectorMapper, self).__init__()
        # Predict genres using the aggregated features.
        self.mapper = nn.Sequential(nn.Linear(vector_dim, 3*vector_dim),
                                    nn.ReLU(),
                                    nn.Linear(3*vector_dim, 5*vector_dim),
                                    nn.ReLU(),
                                    nn.Linear(5 * vector_dim, 3 * vector_dim),
                                    nn.ReLU(),
                                    nn.Linear(3*vector_dim, vector_dim))
        self.current_mode = mode['img2snd']
        self.mode = mode
        
    def forward(self, x, mode):
        if mode != self.current_mode:
            self.current_mode = mode
        x = self.mapper(x)
        return x

    def inverse_layers(self):
        for i in range(len(self.mapper)//2):
            self.inverse_layer(self.mapper[i], self.mapper[len(self.mapper)-i-1],
                               same=(i == len(self.mapper)-i-1))
        self.mapper = nn.Sequential(*list(self.mapper.children()))

    def inverse_layer(self, layer_A, layer_B, debug=False, same=False):
        sd_A = layer_A.state_dict()
        try:
            weight_A, bias_A = sd_A['weight'], sd_A['bias']
            iw_A = torch.pinverse(weight_A, rcond=1e-17)
            ib_A = torch.matmul(iw_A, -1. * bias_A)
            if debug:
                ow_A = torch.pinverse(iw_A)
                ob_A = torch.matmul(ow_A, -1 * ib_A)
                breakpoint()
            if same:
                sd_A['weight'], sd_A['bias'] = iw_A, ib_A
                layer_A.load_state_dict(sd_A)
            else:
                sd_B = layer_B.state_dict()
                weight_B, bias_B = sd_B['weight'], sd_B['bias']

                print(weight_B.shape, weight_A.shape)
                breakpoint()
                iw_B = torch.pinverse(weight_B, rcond=1e-17)
                ib_B = torch.matmul(iw_B, -1. * bias_B)

                sd_A['weight'], sd_A['bias'] = iw_B, ib_B
                layer_A.load_state_dict(sd_A)

                sd_B['weight'], sd_B['bias'] = iw_A, ib_A
                layer_B.load_state_dict(sd_B)

            return None
        except KeyError:
            return None


def error_time_test():
    import time
    import numpy as np
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    map = VectorMapper()
    max_error = 0
    min_error = np.inf
    infer_time = 0.

    for i in range(10):
        # if i % 100 == 0:
        print(i)
        dummy = torch.randn((3000, 128))
        st = time.time()
        out = map(dummy, mode=map.mode['snd2img'])
        infer_time += time.time()-st
        st = time.time()
        recon_dummy = map(out, mode=map.mode['img2snd'])
        infer_time += time.time()-st
        error = torch.abs(100 * torch.sum(dummy - recon_dummy) / torch.sum(dummy))
        if error > max_error:
            max_error = error
        if error < min_error:
            min_error = error
    print(f'Max Error : {max_error/30000}, Min Error: {min_error/30000}, Inference time: {infer_time/60000}%')

def learning_test(dim=30):
    import time
    import numpy as np
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    map = VectorMapper(vector_dim=dim)
    optimizer = torch.optim.SGD(map.parameters(), lr=0.001, weight_decay=1e-2)
    criterion = torch.nn.MSELoss()
    dummy = torch.randn((300000, dim))/2 + 2.5
    target = torch.randn((300000, dim))/4
    for epoch in range(10):
        for i in range(50):
            optimizer.zero_grad()
            out = map(dummy[6000*i: 6000*i+3000], mode=map.mode['snd2img'])
            loss = criterion(out, target[6000*i: 6000*i+3000])
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            out = map(target[6000*i+3000: 6000*i+6000], mode=map.mode['img2snd'])
            loss = criterion(out, dummy[6000*i+3000: 6000*i+6000])
            loss.backward()
            optimizer.step()
        if epoch % 100 == 99:
            print(map.mapper.weight, map.mapper.bias)
    breakpoint()


def learn_mapping(device):
    from tqdm import tqdm
    from sound_classifier.VAE_MMGAN import VAE
    from image_classifier.train_tools.models import ImgVAE

    img_classifier= ImgVAE(z_dim=128, output_units=20)
    checkpoint = torch.load('image_classifier/experiments/2020-12-15-11_08_38/checkpoint_0009.pth.tar',
                            map_location='cpu')

    img_classifier.load_state_dict(checkpoint['state_dict_G'])
    img_classifier = img_classifier.classifier.to(device)

    snd_classifier = VAE(z_dim=128, output_units=20)
    checkpoint = torch.load('sound_classifier/experiments/2020-12-08-15_25_02/checkpoint_0989.pth.tar',
                            map_location=device)
    snd_classifier.load_state_dict(checkpoint['state_dict_G'])
    snd_classifier = snd_classifier.classifier.to(device)

    map = [VectorMapper(vector_dim=128), VectorMapper(vector_dim=128)]
    optimizer = [torch.optim.Adam(map[0].parameters(), lr=3e-4, weight_decay=0.),
                 torch.optim.Adam(map[1].parameters(), lr=3e-4, weight_decay=0.)]
    criterion = [AVContrastLoss(), torch.nn.CrossEntropyLoss(), torch.nn.L1Loss()]
    map[0].to(device)
    map[1].to(device)
    image_dataset = VectorsDataset(mode='image')
    audio_dataset = VectorsDataset(mode='audio')

    image_sampler = BalancedBatchSampler(image_dataset, n_classes=20, n_samples=audio_dataset.__len__()//5)
    audio_sampler = BalancedBatchSampler(audio_dataset, n_classes=20, n_samples=audio_dataset.__len__()//20)

    image_loader = DataLoader(image_dataset, batch_sampler=image_sampler, num_workers=0)
    audio_loader = DataLoader(audio_dataset, batch_sampler=audio_sampler, num_workers=0)
    audio_loader_iter = iter(audio_loader)
    from datetime import datetime
    import os
    now = datetime.now()
    experiment_id = str(now.strftime('%Y-%m-%d-%H_%M_%S'))
    model_folder = 'mapper_experiments_vaes/' + str(experiment_id) + '/'
    os.makedirs(model_folder)

    for epoch in range(100):
        losslist = [0., 0., 0., 0., 0., 0.]
        acclist = [0., 0.]
        itemlist= [0., 0.]
        epoch_loss_s2i = 0.
        epoch_loss_i2s = 0.
        pbar = tqdm(image_loader, desc=f'Train Epoch {epoch:02}')
        for v_features, v_labels in pbar:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            img_classifier.zero_grad()
            snd_classifier.zero_grad()

            try:
                a_features, a_labels = next(audio_loader_iter)
            except StopIteration:
                audio_loader_iter = iter(audio_loader)
                a_features, a_labels = next(audio_loader_iter)
            assert a_features.shape[-1] != 431
            v_features, v_labels, a_features, a_labels = \
                v_features.to(device), v_labels.to(device), a_features.to(device), a_labels.to(device)
            snd2img = map[0](a_features, mode=map[0].mode['snd2img'])
            s2s = map[1](snd2img, mode=map[1].mode['img2snd'])
            identity_loss = criterion[2](a_features, s2s)
            losslist[4] += identity_loss.item()
            contrast_loss = criterion[0](snd2img, a_labels, v_features, v_labels)
            losslist[0] += contrast_loss.item()
            pred_s2i_y = img_classifier(snd2img)
            classification_loss = criterion[1](pred_s2i_y, a_labels)
            acclist[0] += accuracy(pred_s2i_y, a_labels)
            itemlist[0] += a_labels.shape[0]
            losslist[1] += classification_loss.item()
            loss = contrast_loss + classification_loss + identity_loss
            loss.backward()
            epoch_loss_s2i += loss.item()
            optimizer[0].step()
            optimizer[1].step()

            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            img2snd = map[1](v_features, mode=map[1].mode['img2snd'])
            i2i = map[0](img2snd, mode=map[0].mode['snd2img'])
            identity_loss = criterion[2](v_features, i2i)
            losslist[5] += identity_loss.item()
            contrast_loss = criterion[0](img2snd, v_labels, a_features, a_labels, temp=0.21)
            losslist[2] += contrast_loss.item()
            pred_i2s_y = snd_classifier(img2snd)
            classification_loss = criterion[1](pred_i2s_y, v_labels)
            acclist[1] += accuracy(pred_i2s_y, v_labels)
            itemlist[1] += v_labels.shape[0]
            losslist[3] += classification_loss.item()
            loss = contrast_loss + classification_loss# + identity_loss
            loss.backward()
            epoch_loss_i2s += loss.item()
            optimizer[0].step()
            optimizer[1].step()

        print(epoch_loss_s2i/pbar.total, epoch_loss_i2s/pbar.total)
        print(losslist)
        print(acclist[0]/itemlist[0], acclist[1]/itemlist[1])
        # if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict_snd2img': map[0].state_dict(),
            'state_dict_img2snd': map[1].state_dict(),
            'optimizer_snd2img': optimizer[0].state_dict(),
            'optimizer_img2snd': optimizer[1].state_dict(),
            'loss' : (epoch_loss_i2s + epoch_loss_s2i) / 2*pbar.total
         }

        # save checkpoint in appropriate path (new or best)
        save_checkpoint(checkpoint, epoch, model_folder)


def save_checkpoint(state, epoch, checkpoint_dir):
    checkpoint_path = f'{checkpoint_dir}checkpoint_{epoch:04}.pth.tar'
    torch.save(state, checkpoint_path)


def test_mapper(device, resume = 'mapper_experiments/2020-12-14-13_01_21/checkpoint_0040.pth.tar'):
    from tqdm import tqdm
    from image_classifier.train_tools import resnet20
    from sound_classifier.VAE_MMGAN import VAE
    img_classifier = resnet20(dim_in=1, num_classes=20)
    img_classifier.load_state_dict(torch.load('image_classifier/results/res20_0.8982.pth'))
    print(img_classifier.eval())
    img_classifier = img_classifier.classifier.to(device)
    print(img_classifier.eval())
    snd_classifier = VAE(z_dim=128, output_units=20)
    checkpoint = torch.load('sound_classifier/experiments/2020-12-08-15_25_02/checkpoint_0989.pth.tar', map_location=device)
    snd_classifier.load_state_dict(checkpoint['state_dict_G'])
    print(snd_classifier.eval())
    snd_classifier = snd_classifier.classifier.to(device)


    map = [VectorMapper(vector_dim=128), VectorMapper(vector_dim=128)]
    checkpoint = torch.load(resume)
    map[0].load_state_dict(checkpoint['state_dict_snd2img'])
    map[1].load_state_dict(checkpoint['state_dict_img2snd'])
    map[0].to(device)
    map[1].to(device)
    image_dataset = VectorsDataset(mode='image')
    audio_dataset = VectorsDataset(mode='audio')

    image_loader = DataLoader(image_dataset, batch_size=20000, shuffle=False, drop_last=False)
    audio_loader = DataLoader(audio_dataset, batch_size=256, shuffle=False, drop_last=False)

    pbar = tqdm(audio_loader, desc=f'Val snd2img')
    acc = 0.
    for a_features, a_labels in pbar:
        snd2img = map[0](a_features.to(device), mode=map[0].mode['snd2img'])
        pred_y = img_classifier(snd2img)
        acc += accuracy(pred_y.cpu(), a_labels)
    print(f'snd2img accuracy : {acc/audio_dataset.__len__()}')

    pbar = tqdm(image_loader, desc=f'Val img2snd')
    acc = 0.
    for v_features, v_labels in pbar:
        img2snd = map[1](v_features.to(device), mode=map[1].mode['img2snd'])
        pred_y = snd_classifier(img2snd)
        acc += accuracy(pred_y.cpu(), v_labels)
    print(f'snd2img accuracy : {acc/image_dataset.__len__()}')
    breakpoint()


def accuracy(source, target):
    source = source.max(1)[1].long().cpu()
    target = target.cpu()
    correct = (source == target).sum().item()
    return correct



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn', dest='learn',action='store_true',
                        help='Use MusiCNN Front-End Mid-End architecture for Discriminator')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Use MusiCNN Front-End Mid-End architecture for Discriminator')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import argparse
    args = parse()
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.learn:
        learn_mapping(device)
    if args.test:
        if args.resume is None:
            test_mapper(device)
        else:
            test_mapper(device, args.resume)
