from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models

vgg = models.vgg19().features

vgg.load_state_dict(torch.load(
    'features.pth', map_location="cuda" if torch.cuda.is_available() else "cpu"))


for param in vgg.parameters():
    param.requires_grad_(False)


for i, layer in enumerate(vgg):
    if isinstance(layer, torch.nn.MaxPool2d):
        vgg[i] = torch.nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0)


class nst:

    def __init__(self, content_path, style_path, save_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.content = self.load_image(content_path).to(self.device)
        self.style = self.load_image(style_path).to(self.device)

        self.save_path = save_path

        self.model = vgg.to(self.device).eval()
        self.style_weights = {'conv1_1': 0.75,
                              'conv2_1': 0.5,
                              'conv3_1': 0.3,
                              'conv4_1': 0.3,
                              'conv5_1': 0.3}
        self.target = torch.randn_like(
            self.content).requires_grad_(True).to(self.device)
        self.content_weight = 1e5
        self.style_weight = 1e2
        self.optimizer = optim.Adam([self.target], lr=0.01)

    def load_image(self, img_path, max_size=512, shape=None):
        image = Image.open(img_path).convert('RGB')

        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose([
            transforms.Resize((size, int(size))),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        image = in_transform(image)[:3, :, :].unsqueeze(0)

        return image

    def im_convert(self):
        image = self.target.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array(
            (0.485, 0.456, 0.406))
        image = image.clip(0, 1)*255

        return Image.fromarray(image.astype(np.uint8))

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',  # content layer
                      '28': 'conv5_1'}
        features = {}
        x = image
        for name, layer in enumerate(self.model):
            x = layer(x)
            if str(name) in layers:
                features[layers[str(name)]] = x

        return features

    def gram_matrix(self, tensor):
        _, n_filters, h, w = tensor.size()
        tensor = tensor.view(n_filters, h * w)
        gram = torch.mm(tensor, tensor.t())

        return gram

    def predict(self):
        content = self.content.to(self.device)
        style = self.style.to(self.device)

        content_features = self.get_features(content)
        style_features = self.get_features(style)

        style_grams = {
            layer: self.gram_matrix(style_features[layer]) for layer in style_features}

        #target = torch.randn_like(content).requires_grad_(True).to(self.device)

        for i in range(1, 151):
            self.optimizer.zero_grad()
            target_features = self.get_features(self.target.clone())

            content_loss = torch.mean((target_features['conv4_2'] -
                                      content_features['conv4_2']) ** 2)

            style_loss = 0
            for layer in self.style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = self.style_weights[layer] * torch.mean(
                    (target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (d * h * w)

                total_loss = self.content_weight * content_loss + self.style_weight * style_loss
                total_loss.backward(retain_graph=True)
                self.optimizer.step()
        res_image = self.im_convert()
        res_image.save(self.save_path)
