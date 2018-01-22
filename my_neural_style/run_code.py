import torch
from my_neural_style import build_model


def get_input_param_and_optimier(input_img):
    input_param = torch.nn.Parameter(input_img.data)
    optimier = torch.optim.LBFGS([input_param])
    return input_param, optimier


def run_style_transfer(content_img, style_image, input_img, num_epochs=300):
    model, style_loss_list, content_loss_list = build_model.get_style_model_and_loss(style_image, content_img)
    input_param, optimizer = get_input_param_and_optimier(input_img)

    epoch = [0]
    while epoch[0] < num_epochs:
        def closure():
            # 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。
            #       | min, if x_i < min
            # y_i = | x_i, if min <= x_i <= max
            #       | max, if x_i > max
            input_param.data.clamp_(0, 1)
            model(input_param)
            content_score = 0
            style_score = 0
            optimizer.zero_grad()
            for cl in content_loss_list:
                content_score += cl.backward()

            for sl in style_loss_list:
                style_score += sl.backward()
            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run{}'.format(epoch))
                print('Style Loss:{:.4f} Content Loss{:.4f}'.format(style_score.data[0], content_score.data[0]))


            return style_score + content_score

        optimizer.step(closure)
        input_param.data.clamp_(0, 1)
    return input_param.data
