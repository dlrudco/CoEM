import cv2
from final_model import FinalModel
from image_classifier.data_utils import *
import torch
from sound_classifier.dataset import SpecDataset
import matplotlib.pyplot as plt
from librosa import display
import os
import numpy as np
import pickle

dataloaders, dataset_sizes = quickdraw_setter(root='Data/quickdraw', batch_size=1, num_workers=0)
train_dataset = SpecDataset(root='Data/umsun_dataset_44k/spec', mode='train')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FinalModel(device)
model.eval()
count = {}
tcount = 0

a_dict={}
a_dict['paths'] = pickle.load(open('sound_classifier/a_paths.pkl', 'rb'))
a_dict['vectors']= np.load('sound_classifier/a_features.npy')
a_dict['labels'] = np.load('sound_classifier/a_labels.npy')

a_dict['s2i_vectors'] = torch.from_numpy(a_dict['vectors'])
a_dict['norm_vectors'] = torch.nn.functional.normalize(a_dict['s2i_vectors'], dim=1)


with torch.no_grad():
    for images, v_labels in dataloaders['train']:
        tcount += 1
        gen_sound, cls, i2s = model(images.to(device), prior=v_labels.to(device), mode='i2s', return_feature=True)
        sim = torch.einsum('ak, vk -> av',
                           torch.nn.functional.normalize(i2s, dim=1).to(device),
                           a_dict['norm_vectors'].to(device))
        # sim_max = torch.max(sim, dim=1, keepdim=True)
        topk = torch.topk(sim, 20, dim=1)
        frequent = torch.mode(torch.from_numpy(a_dict['labels'][topk[1][0].cpu()]))
        ret_index = topk[1][0, frequent[1]]
        ret_path = a_dict['paths'][ret_index]
        ret_spec = cv2.imread(ret_path.replace('.wav', '.jpg').replace('wav', 'images').replace('../', ''))
        print(tcount, v_labels[0].item(), cls[0].item(), a_dict['labels'][ret_index], ret_path.replace('.wav', '.jpg').replace('wav', 'images').replace('../', ''))
        if v_labels[0].item() == cls[0].item() and v_labels[0].item() == a_dict['labels'][ret_index]:
            if v_labels[0].item() in count.keys():
                count[v_labels[0].item()] = count[v_labels[0].item()] + 1
            else:
                count[v_labels[0].item()] = 1

            if count[v_labels[0].item()] < 20:
                print(v_labels[0].item(), '-', count[v_labels[0].item()])
                os.makedirs(f'result_final_i2s/{v_labels[0].item()}', exist_ok=True)
                temp = (images.cpu()/0.3345 + 0.1767).numpy()
                temp -= temp.min()
                ratio = 255. / temp.max()
                temp = temp * ratio
                temp.astype(np.uint8)
                cv2.imwrite(f'result_final_i2s/{v_labels[0].item()}/{count[v_labels[0].item()]}_input_img.jpg', temp[0,0,:,:])
                pred_x = gen_sound.cpu().numpy()[0, 0, :, :]
                pred_mean, pred_std = pred_x.mean(), pred_x.std()
                pred_x = (pred_x - pred_mean) / pred_std
                # pred_x = pred_x * train_dataset.std + train_dataset.mean
                pred_x = 100*(pred_x - pred_x.min())/(pred_x.max()-pred_x.min()) - 50.
                pred_x = cv2.medianBlur(pred_x, ksize=5)
                pred_x = cv2.medianBlur(pred_x, ksize=3)
                plt.figure(figsize=(10, 4))
                display.specshow(pred_x, y_axis='mel', sr=44100, hop_length=1024, x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel-Spectrogram')
                plt.tight_layout()
                plt.savefig(f'result_final_i2s/{v_labels[0].item()}/{count[v_labels[0].item()]}_gen_spec.jpg')
                plt.close()
                cv2.imwrite(f'result_final_i2s/{v_labels[0].item()}/{count[v_labels[0].item()]}_orig_spec.jpg', ret_spec)