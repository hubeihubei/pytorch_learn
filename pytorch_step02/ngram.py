import torch
import torch.nn as nn
from torch.autograd import Variable


def embedding01():
    word_to_idx = {'hello': 0, 'word': 1}
    # 定义词嵌入
    embeds = nn.Embedding(2, 5)  # 2 个单词，维度 5
    hello_index = torch.LongTensor([word_to_idx['hello']])
    hello_index = Variable(hello_index)
    # 访问第 1 个词的词向量
    hello_embedding = embeds(hello_index)
    print('hello_embedding', hello_embedding)
    # 得到词嵌入矩阵
    print('embeds.weight', embeds.weight)


# embedding01()
# 依据的单词个数
CONTEXT_SIZE = 2
# 词向量的维度
EMBEDDING_DIM = 20
test_sentence = '''In my dual profession as an educator and health care provider, I have worked with numerous children infected with the virus that causes AIDS. The relationships that I have had with these special kids have been gifts in my life. They have taught me so many things, but I have especially learned that great courage can be found in the smallest of packages. Let me tell you about Tyler.

Tyler was born infected with HIV: his mother was also infected. From the very beginning of his life, he was dependent on medications to enable him to survive. When he was five, he had a tube surgically inserted in a vein in his chest. This tube was connected to a pump, which he carried in a small backpack on his back. Medications were hooked up to this pump and were continuously supplied through this tube to his bloodstream. At times, he also needed supplemented oxygen to support his breathing.

Tyler wasn’t willing to give up one single moment of his childhood to this deadly disease. It was not unusual to find him playing and racing around his backyard, wearing his medicine-laden backpack and dragging his tank of oxygen behind him in his little wagon. All of us who knew Tyler marveled at his pure joy in being alive and the energy it gave him. Tyler’s mom often teased him by telling him that he moved so fast she needed to dress him in red. That way, when she peered through the window to check on him playing in the yard, she could quickly spot him.

This dreaded disease eventually wore down even the likes of a little dynamo like Tyler. He grew quite ill and, unfortunately, so did his HIV-infected mother. When it became apparent that he wasn’t going to survive, Tyler’s mom talked to him about death. She comforted him by telling Tyler that she was dying too, and that she would be with him soon in heaven.

A few days before his death, Tyler beckoned me over to his hospital bed and whispered, “I might die soon. I’m not scared. When I die, please dress me in red. Mom promised she’s coming to heaven, too. I’ll be playing when she gets there, and I want to make sure she can find me.”'''.split()
# print(test_sentence)
# 这里的 CONTEXT_SIZE 表示我们希望由前面几个单词来预测这个单词，这里使用两个单词，EMBEDDING_DIM 表示词嵌入的维度。
# 接着我们建立训练集，便利整个语料库，将单词三个分组，前面两个作为输入，最后一个作为预测的结果。
trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2]) for i in range(test_sentence.__len__() - 2)]
# print(trigram)

# length 381
# print(len(trigram))

# length 229
vocb = set(test_sentence)
# print(len(vocb))

train_data=trigram[:300]
test_data=trigram[300:]


# 建立每个词与数字的编码，据此构建词嵌入
word_to_idx = {word: i for i, word in enumerate(vocb)}
# print(word_to_idx)
# 建立每个数字与词的编码
idx_to_word = {i: word for i, word in enumerate(vocb)}
# print(idx_to_word)
vocb_size = len(vocb)


class NGRAM(nn.Module):
    def __init__(self):
        super(NGRAM, self).__init__()
        self.embed = nn.Embedding(vocb_size, EMBEDDING_DIM)
        self.drop1=nn.Dropout(0.4)
        self.classifier = nn.Sequential(
            nn.Linear(CONTEXT_SIZE * EMBEDDING_DIM, 300),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(300, vocb_size),
        )

    def forward(self, x):
        x = self.embed(x)
        # x会是一个vocb_size,EMBEDDING_DIM (2,10)的矩阵，需要展开为（1,2*10）
        # print(x.size())
        x = x.view(1, -1)
        x = self.classifier(x)
        return x

ngram=NGRAM().cuda()
optimization=torch.optim.Adam(ngram.parameters(),0.001)
loss_func=nn.CrossEntropyLoss().cuda()
for i in range(700):
    train_loss=0
    for word,label in trigram:
        word=Variable(torch.LongTensor([word_to_idx[i] for i in word])).cuda()
        label=Variable(torch.LongTensor([word_to_idx[label]])).cuda()
        out=ngram(word)
        loss=loss_func(out,label)
        train_loss+=loss.data[0]
        optimization.zero_grad()
        loss.backward()
        optimization.step()

    if i% 100==0 or i==499:
        print("loss:",train_loss/len(trigram))

pre_list=[]
real_list=[]
ngram.eval()

# 因为数据量太少所以准确率很低
for word_test,label_test in test_data:
    word_test=Variable(torch.LongTensor([word_to_idx[i] for i in word_test])).cuda()
    out=ngram(word_test)
    prediction_index=torch.max(out,1)[1].cuda().data  # type LongTensor
    # print(prediction_index)
    prediction_index=prediction_index[0] # type long
    # print(prediction_index)
    prediction=idx_to_word[prediction_index]
    pre_list.append(prediction)
    real_list.append(label_test)

print("real:",real_list)
print("pred:",pre_list)