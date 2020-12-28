import cv2
from final_model import FinalModel
import torch
import matplotlib.pyplot as plt
from librosa import display
import os
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

v_dict={}
v_dict['images'] = np.load('image_classifier/v_images.npy')
v_dict['vectors'] = np.load('image_classifier/v_features.npy')
v_dict['labels'] = np.load('image_classifier/v_labels.npy')
a_dict={}

a_dict['paths'] = pickle.load(open('sound_classifier/a_paths.pkl','rb'))
a_dict['vectors']= np.load('sound_classifier/a_features.npy')
a_dict['labels'] = np.load('sound_classifier/a_labels.npy')

model = FinalModel(device)

mapper_i2s = model.mapper_img2snd
mapper_s2i = model.mapper_snd2img

v_dict['i2s_vectors'] = torch.from_numpy(v_dict['vectors'])
v_dict['norm_vectors'] = torch.nn.functional.normalize(v_dict['i2s_vectors'], dim=1)
v_dict['i2s_vectors'] = mapper_i2s(v_dict['i2s_vectors'].to(device), mode=mapper_i2s.mode['img2snd'])
v_dict['i2s_vectors'] = torch.nn.functional.normalize(v_dict['i2s_vectors'], dim=1)


a_dict['s2i_vectors'] = torch.from_numpy(a_dict['vectors'])
a_dict['norm_vectors'] = torch.nn.functional.normalize(a_dict['s2i_vectors'], dim=1)
a_dict['s2i_vectors'] = mapper_s2i(a_dict['s2i_vectors'].to(device), mode=mapper_s2i.mode['snd2img'])
a_dict['s2i_vectors'] = torch.nn.functional.normalize(a_dict['s2i_vectors'], dim=1)

model.eval()
count = {}


with torch.no_grad():
    for i in range(a_dict['s2i_vectors'].shape[0]):
        spec = a_dict['paths'][i]
        spec = spec.replace('.wav', '.npy').replace('wav', 'spec').replace('../', '')
        spec = np.load(spec)
        spec = (spec + 36.323814) / 20.011667
        spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        a_label = a_dict['labels'][i]
        a_label = torch.from_numpy(np.expand_dims(a_label, axis=0))

        gen_image, cls, s2i = model(spec.to(device), prior=a_label.to(device), mode='s2i', return_feature=True)
        sim = torch.einsum('ak, vk -> av',
                           torch.nn.functional.normalize(s2i, dim=1).to(device),
                           v_dict['norm_vectors'].to(device))
        topk = torch.topk(sim, 20, dim=1)
        frequent = torch.mode(torch.from_numpy(v_dict['labels'][topk[1][0].cpu()]))
        ret_index = topk[1][0, frequent[1]]
        ret_img = v_dict['images'][ret_index]
        print(i, a_label.item(), cls[0].item(), v_dict['labels'][ret_index])
        if a_label.item() == cls[0].item() and a_label.item() == v_dict['labels'][ret_index]:
            if a_label.item() in count.keys():
                count[a_label.item()] = count[a_label.item()] + 1
            else:
                count[a_label.item()] = 1

            if count[a_label.item()] < 20:
                print(a_label.item(), '-', count[a_label.item()])
                os.makedirs(f'result_final_s2i/{a_label.item()}', exist_ok=True)
                pred_x = spec
                pred_x = pred_x * 20.011667 - 36.323814

                plt.figure(figsize=(10, 4))
                display.specshow(pred_x.cpu().numpy()[0,0,:,:], y_axis='mel', sr=44100, hop_length=1024, x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel-Spectrogram')
                plt.tight_layout()
                plt.savefig(f'result_final_s2i/{a_label.item()}/{count[a_label.item()]}_input_spec.jpg')
                plt.close()

                temp = (gen_image.cpu()/0.3345 + 0.1767).numpy()
                temp -= temp.min()
                ratio = 255. / temp.max()
                temp = temp * ratio
                temp.astype(np.uint8)
                cv2.imwrite(f'result_final_s2i/{a_label.item()}/{count[a_label.item()]}_gen_img.jpg', temp[0,0,:,:])
                ret_img = ret_img[0] / 0.3345 + 0.1767
                ret_img -= ret_img.min()
                ratio = 255. / ret_img.max()
                ret_img = ret_img * ratio
                ret_img.astype(np.uint8)
                cv2.imwrite(f'result_final_s2i/{a_label.item()}/{count[a_label.item()]}_ret_img.jpg', ret_img)
