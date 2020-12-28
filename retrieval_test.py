from final_model import FinalModel
import torch
import numpy as np
import pickle


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--topk', default=20, type=int,
                        help='use k nearest neighbors for test')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import argparse
    args = parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    v_dict={}
    # v_dict['images'] = np.load('image_classifier/images.npy')
    v_dict['vectors'] = np.load('image_classifier/v_features.npy')
    v_dict['labels'] = np.load('image_classifier/v_labels.npy')
    a_dict={}

    a_dict['paths'] = pickle.load(open('sound_classifier/a_paths.pkl','rb'))
    a_dict['vectors']= np.load('sound_classifier/a_features.npy')
    a_dict['labels'] = np.load('sound_classifier/a_labels.npy')

    model = FinalModel(device, args.resume)
    model.eval()

    mapper_i2s = model.mapper_img2snd
    mapper_s2i = model.mapper_snd2img
    mapper_i2s.eval()
    mapper_s2i.eval()
    v_dict['i2s_vectors'] = torch.from_numpy(v_dict['vectors'])
    v_dict['norm_vectors'] = torch.nn.functional.normalize(v_dict['i2s_vectors'], dim=1)
    v_dict['i2s_vectors'] = mapper_i2s(v_dict['i2s_vectors'].to(device), mode=mapper_i2s.mode['img2snd'])
    v_dict['i2s_vectors'] = torch.nn.functional.normalize(v_dict['i2s_vectors'], dim=1)


    a_dict['s2i_vectors'] = torch.from_numpy(a_dict['vectors'])
    a_dict['norm_vectors'] = torch.nn.functional.normalize(a_dict['s2i_vectors'], dim=1)
    a_dict['s2i_vectors'] = mapper_s2i(a_dict['s2i_vectors'].to(device), mode=mapper_s2i.mode['snd2img'])
    a_dict['s2i_vectors'] = torch.nn.functional.normalize(a_dict['s2i_vectors'], dim=1)
    a_path = []
    breakpoint()
    correct_A = 0
    for i in range(v_dict['i2s_vectors'].shape[0]):
        sim = torch.einsum('ak, vk -> av',
                           v_dict['i2s_vectors'][i].unsqueeze(0).to(device),
                           a_dict['norm_vectors'].to(device))
        topk = torch.topk(sim, args.topk, dim=1)
        frequent = torch.mode(torch.from_numpy(np.expand_dims(a_dict['labels'][topk[1][0]], axis=0)))
        ret_index = topk[1][0, frequent[1]]
        if v_dict['labels'][i] == a_dict['labels'][ret_index]:
            correct_A += 1
            # a_path.append(v_dict['images'][i])
    print(100*correct_A/i)

    b_path = []
    correct_B = 0
    for i in range(a_dict['s2i_vectors'].shape[0]):
        if i % 100 == 0:
            print(i, correct_B/(i+1))
        sim = torch.einsum('ak, vk -> av',
                           a_dict['s2i_vectors'][i].unsqueeze(0).to(device),
                           v_dict['norm_vectors'].to(device))
        topk = torch.topk(sim, args.topk, dim=1)
        frequent = torch.mode(torch.from_numpy(np.expand_dims(v_dict['labels'][topk[1][0]], axis=0)))
        ret_index = topk[1][0, frequent[1]]
        if a_dict['labels'][i] == v_dict['labels'][ret_index]:
            correct_B += 1
            b_path.append(a_dict['paths'][i])
    print(100*correct_B/i)
    breakpoint()