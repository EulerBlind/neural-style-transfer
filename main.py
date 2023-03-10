from __future__ import print_function

from torch import optim
from torchvision import models

from image_util import *
from model_util import *


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers):
    # 正则化，将图片的像素值归一化到[-1,1]
    # 由于vgg训练的时候，图片的像素值也是归一化到[-1,1]的，所以这里也要归一化
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img: torch.Tensor):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def main_style_transfer(cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps=300,
                        style_weight=1000000, content_weight=1000,
                        content_layers=None, style_layers=None):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean, normalization_std, style_img,
        content_img,
        content_layers=content_layers,
        style_layers=style_layers
    )
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss: torch = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}\n'.format(
                    style_score.item(), content_score.item()))
                imshow(input_img, title=f'Output Image {run[0]}')
            return style_score + content_score

        optimizer.step(closure)
    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == '__main__':
    content_image, style_image = load_content_and_style_image("aaa.jpg", "bbb.png")
    imshow(content_image, title="Content Image")
    imshow(style_image, title="Style Image")
    model = models.vgg19(pretrained=True).features.to(device).eval()

    # 这里给定这些值，是因为vgg19训练的时候，图片的像素值是归一化到[-1,1]的
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # 定义自己的层，这里完全是根据自己认定来判别的
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    input_img = content_image.clone()
    # input_img = torch.rand(content_image.shape, device=device, requires_grad=True)

    output = main_style_transfer(model, cnn_normalization_mean, cnn_normalization_std,
                                 content_image, style_image, input_img,
                                 content_layers=content_layers,
                                 style_layers=style_layers)

    imshow(output, title='Output Image')
