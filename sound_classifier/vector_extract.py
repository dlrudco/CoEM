from dataset import SpecDataset
from VAE_MMGAN import VAE
import numpy as np
import torch
import argparse

def ext_vectors(args):
    device = args.device

    train_dataset = SpecDataset(mode='train')

    model = VAE(z_dim=128, output_units=20)
    checkpoint = torch.load('experiments/checkpoint_0989.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict_G'])
    model.to(device)
    print(model.eval())
    paths = []
    vectors, vector_labels = None, None
    for i, (path, labels) in enumerate(train_dataset.paths):
        print(path)
        spec = np.load(path)
        spec = spec[:, :train_dataset.time_dim_size]
        spec = (spec - train_dataset.mean) / train_dataset.std
        spec = np.expand_dims(np.expand_dims(spec, axis=0), axis=0)
        features = torch.from_numpy(spec)
        labels = torch.from_numpy(np.array(labels)).long()
        X = features.to(device=device)
        y = labels.to(device=device)
        with torch.no_grad():
            y_onehot = torch.zeros((1, 20))
            y_onehot[0, y.item()] = 1.
            pred_x, pred_z, mu, logvar, _y, feature = model(X, is_training=True, prior=y_onehot.to(device), original_feature=True)
            paths.append(path)
            vectors = tensor_concater(vectors, feature.cpu())
            vector_labels = tensor_concater(vector_labels, y.cpu().unsqueeze(0))
    import pickle
    pickle.dump(paths, open('a_paths.pkl', 'wb'))
    np.save('./a_features.npy', vectors.numpy())
    np.save('./a_labels.npy', vector_labels.numpy())

def tensor_concater(tensor1, tensor2):
    if tensor1 is None:
        tensor1 = tensor2
    else:
        tensor1 = torch.cat((tensor1, tensor2), dim=0)
    return tensor1


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def parse():
    parser = argparse.ArgumentParser(description='CoEM vector extraction script')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ext_vectors(args)