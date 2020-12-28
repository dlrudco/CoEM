import torch
from torch import nn
from image_classifier.train_tools.models.VAE_MMGAN import ImgVAE
from sound_classifier.VAE_MMGAN import VAE
from vector_mapping import VectorMapper

class FinalModel(nn.Module):
    def __init__(self, device, mapper_path=None):
        super(FinalModel, self).__init__()
        self.image_encoder, self.image_decoder, self.image_classifier, self.image_reparam = self.build_image_generator()
        self.sound_encoder, self.sound_decoder, self.sound_classifier, self.sound_reparam = self.build_sound_generator()
        self.mapper_img2snd = VectorMapper()
        self.mapper_snd2img = VectorMapper()
        self.init_mapper(mapper_path)
        self.available_modes = ['i2s', 's2i', 's2s', 'i2i']
        self.device = device
        self.set_device(device)

    def forward(self, x, mode, prior=None, return_feature=False):
        if mode not in self.available_modes:
            raise ValueError

        if mode == 'i2s':
            mu, logvar, img_embed = self.image_encoder(x)
            img_embed = img_embed.view(img_embed.shape[0], -1)
            img2snd = self.mapper_img2snd(img_embed, mode=self.mapper_img2snd.mode['img2snd'])
            reparam = self.sound_reparam(img2snd)
            mu = reparam[:, :128]
            logvar = reparam[:, :128]
            z = self.reparameterize(mu, logvar)
            cls = self.sound_classifier(img2snd)
            cls = cls.max(1)[1]
            y_onehot = torch.zeros((cls.shape[0], 20)).to(self.device)
            if prior is None:
                for i in range(cls.shape[0]):
                    y_onehot[0, cls[i].item()] = 1.
                gen_sound = self.sound_decoder(torch.cat((img2snd, y_onehot, z), dim=1))
            else:
                for i in range(prior.shape[0]):
                    y_onehot[0, prior[i].item()] = 1.
                gen_sound = self.sound_decoder(torch.cat((img2snd, y_onehot, z), dim=1))

            if return_feature:
                return gen_sound, cls, img_embed
            else:
                return gen_sound, cls

        elif mode == 's2i':
            mean, logvar, snd_embed = self.sound_encoder(x)
            snd_embed = snd_embed.view(snd_embed.shape[0], -1)
            snd2img = self.mapper_snd2img(snd_embed, mode=self.mapper_snd2img.mode['snd2img'])
            reparam = self.image_reparam(snd2img)
            mu = reparam[:, :128]
            logvar = reparam[:, :128]
            z = self.reparameterize(mu, logvar)
            cls = self.image_classifier(snd2img)
            cls = cls.max(1)[1]
            y_onehot = torch.zeros((cls.shape[0], 20)).to(self.device)
            if prior is None:
                for i in range(cls.shape[0]):
                    y_onehot[0, cls[i].item()] = 1.
                gen_image = self.image_decoder(torch.cat((snd2img, y_onehot, z), dim=1))
            else:
                for i in range(prior.shape[0]):
                    y_onehot[0, prior[i].item()] = 1.
                gen_image = self.image_decoder(torch.cat((snd2img, y_onehot, z), dim=1))

            if return_feature:
                return gen_image, cls, snd_embed
            else:
                return gen_image, cls

        elif mode == 'i2i':
            mu, logvar, img_embed = self.image_encoder(x)
            z = self.reparameterize(mu, logvar)
            cls = self.image_classifier(img_embed)
            y_onehot = torch.zeros((cls.shape[0], 20)).to(self.device)
            if prior is None:
                for i in range(cls.shape[0]):
                    y_onehot[0, cls[i].item()] = 1.
                gen_image = self.image_decoder(torch.cat((img_embed, y_onehot, z), dim=1))
            else:
                for i in range(prior.shape[0]):
                    y_onehot[0, prior[i].item()] = 1.
                gen_image = self.image_decoder(torch.cat((img_embed, y_onehot, z), dim=1))
            return gen_image, cls
        
        elif mode == 's2s':
            mu, logvar, snd_embed = self.sound_encoder(x)
            z = self.reparameterize(mu, logvar)
            cls = self.sound_classifier(snd_embed)
            y_onehot = torch.zeros((cls.shape[0], 20)).to(self.device)
            if prior is None:
                for i in range(cls.shape[0]):
                    y_onehot[0, cls[i].item()] = 1.
                gen_sound = self.sound_decoder(torch.cat((snd_embed, y_onehot, z), dim=1))
            else:
                for i in range(prior.shape[0]):
                    y_onehot[0, prior[i].item()] = 1.
                gen_sound = self.sound_decoder(torch.cat((snd_embed, y_onehot, z), dim=1))
            return gen_sound, cls

    def build_image_generator(self):
        image_generator = ImgVAE(z_dim=128, output_units=20)
        checkpoint = torch.load('image_classifier/experiments/checkpoint_0009.pth.tar',
                                map_location='cpu')
        image_generator.load_state_dict(checkpoint['state_dict_G'])
        encoder = image_generator.encoder
        decoder = image_generator.decoder
        classifier = image_generator.classifier
        reparam = encoder.vae_fc
        return encoder, decoder, classifier, reparam

    def build_sound_generator(self):
        sound_generator = VAE(z_dim=128, output_units=20)
        checkpoint = torch.load('sound_classifier/experiments/checkpoint_0989.pth.tar',
                                map_location='cpu')
        sound_generator.load_state_dict(checkpoint['state_dict_G'])
        encoder = sound_generator.encoder
        decoder = sound_generator.decoder
        classifier = sound_generator.classifier
        reparam = encoder.vae_fc
        return encoder, decoder, classifier, reparam

    def init_mapper(self, mapper_path):
        if mapper_path is None:
            checkpoint = torch.load('mapper_experiments_vaes/checkpoint_0006.pth.tar',
                                map_location='cpu')
            self.mapper_snd2img.load_state_dict(checkpoint['state_dict_snd2img'])
            self.mapper_img2snd.load_state_dict(checkpoint['state_dict_img2snd'])
        else:
            checkpoint = torch.load(mapper_path,
                                    map_location='cpu')
            self.mapper_snd2img.load_state_dict(checkpoint['state_dict_snd2img'])
            self.mapper_img2snd.load_state_dict(checkpoint['state_dict_img2snd'])
        print('Mapper load done')

    def set_device(self, device):
        self.device = device
        self.image_encoder.to(self.device)
        self.image_decoder.to(self.device)
        self.image_classifier.to(self.device)
        self.image_reparam.to(self.device)
        self.sound_encoder.to(self.device)
        self.sound_decoder.to(self.device)
        self.sound_classifier.to(self.device)
        self.sound_reparam.to(self.device)
        self.mapper_img2snd.to(self.device)
        self.mapper_snd2img.to(self.device)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
