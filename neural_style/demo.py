from torch.autograd import Variable
from torchvision import transforms
from neural_style.run_code import run_style_transfer
from neural_style.load_img import load_img, show_img
from torch.autograd import Variable

style_img = load_img('./picture/style.png')
style_img = Variable(style_img).cuda()
content_img = load_img('./picture/content.png')
content_img = Variable(content_img).cuda()

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)

show_img(out.cpu())