import torchvision.transforms as trans
import PIL.Image as image

img_size = 512


def load_img(img_path):
    img = image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = trans.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def show_img(img):
    img = img.squeeze(0)
    img = trans.ToPILImage()(img)
    img.show()
