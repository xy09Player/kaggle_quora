# encoding = utf-8
# author = xy

from process_data import *
from utils import *
from sklearn import model_selection
from sklearn import metrics
from config import param
from model import model_rnn
import torch
from torch import optim
from torch import nn
import time
import copy


def train():
    # 参数初始化
    train_file = 'data/train.csv'
    embedding_file = 'data/glove.840B.300d.txt'
    max_len = 100
    batch_size = 32

    # 载入数据
    train_questions, train_targets = deal_data(train_file, max_len=max_len)
    w2i, embedding = build_word_embedding(train_questions, embedding_file)

    # index, padding
    train_questions = word2indexs(train_questions, w2i)
    train_questions = padding(train_questions, max_len)

    # split train, val set
    train_questions, val_questions, train_targets, val_targets = model_selection.train_test_split(
        train_questions, train_targets, test_size=0.1, random_state=333)

    assert len(train_questions) == len(train_targets)
    assert len(val_questions) == len(val_targets)
    print('train size:%d, val size:%d' % (len(train_questions), len(val_questions)))

    # build dataloader
    train_loader = get_dataloader(
        dataset=[train_questions, train_targets],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = get_dataloader(
        dataset=[val_questions, val_targets],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # set param
    param['embedding'] = embedding

    # model
    model = model_rnn.Model(param)
    model = model.cuda()
    model_best_state = None
    loss_best = 999

    # loss
    criterion = torch.nn.BCELoss()

    # optimizer
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(optimizer_param, lr=1e-4)

    # train
    model_param_num = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            model_param_num += parameter.nelement()
    print('start training, param_num:%d' % model_param_num)

    train_loss = 0
    train_c = 0
    t_nums = len(train_questions) // batch_size
    every_nums = t_nums // 10
    time0 = time.time()
    for e in range(param['epochs']):
        for i, batch in enumerate(train_loader):
            batch = [b.cuda() for b in batch]
            model.train()
            optimizer.zero_grad()
            outputs = model(batch)
            loss_value = criterion(outputs, batch[1].view(-1, 1).float())
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()
            train_c += 1

            if train_c % every_nums == 0:
                val_loss = 0
                val_c = 0
                correct_num = 0
                sum_num = 0
                with torch.no_grad():
                    model.eval()
                    for val_batch in val_loader:
                        val_batch = [b.cuda() for b in val_batch]
                        outputs = model(val_batch)
                        loss_value = criterion(outputs, val_batch[1].view(-1, 1).float())

                        correct_num += ((outputs > 0.5).long() == val_batch[1].view(-1, 1)).sum().item()
                        sum_num += outputs.size(0)

                        val_loss += loss_value.item()
                        val_c += 1
                print('training, epochs:%2d, steps:%2d/%2d, train_loss:%.4f, val_loss:%.4f, accuracy:%.4f, time:%4ds' %
                      (e, (i+1), t_nums, train_loss/train_c, val_loss/val_c, correct_num/sum_num, time.time()-time0))

                if loss_best > (val_loss / val_c):
                    loss_best = val_loss / val_c
                    model_best_state = copy.deepcopy(model.state_dict())
    print('training, best_loss:%.4f' % loss_best)

    # 确定选择阈值
    model = model_rnn.Model(param)
    model = model.cuda()
    model.load_state_dict(model_best_state)
    model.eval()
    scores = np.arange(0, 1, 0.05)
    best_score = -1
    best_accuracy = 0
    for score in scores:
        correct_num = 0
        sum_num = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = [b.cuda() for b in val_batch]
                outputs = model(val_batch)
                correct_num += ((outputs > score).long() == val_batch[1].view(-1, 1)).sum().item()
                sum_num += outputs.size(0)
        print('score choosing, score:%.2f, accuracy:%.4f' % (score, correct_num/sum_num))
        if best_accuracy < correct_num / sum_num:
            best_score = score
            best_accuracy = correct_num / sum_num
    print('valing, best_score:%.2f, best_accuracy:%.4f' % (best_score, best_accuracy))






    # 预测
    # test数据处理
    test_file = 'data/test.csv'
    test_questions = deal_data(test_file, is_train=False)
    w2i_test, embedding_test = build_word_embedding(test_questions, embedding_file)
    test_questions = word2indexs(test_questions, w2i_test)
    test_questions = padding(test_questions, max_len)
    test_loader = get_dataloader(
        dataset=[test_questions],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # 测试模型构建
    model = model_rnn.Model(param)
    model.load_state_dict(model_best_state)
    model.embedding.embedding_fix = nn.Embedding(
        num_embeddings=embedding_test.shape[0],
        embedding_dim=embedding_test.shape[1],
        padding_idx=0,
        _weight=torch.Tensor(embedding_test)
    )
    model.embedding.vocab_size = embedding_test.shape[0]
    model = model.cuda()

    # 结果生成
    result = []
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = [b.cuda() for b in test_batch]
            outputs = model(test_batch)
            outputs = (outputs > best_score).long()
            result.append(outputs.view(-1).cpu().numpy().tolist())

    # 存储
    test_df = pd.read_csv(test_file)
    submission = pd.DataFrame(
        {'qid': test_df['qid'], 'prediction': result},
        columns=['qid', 'prediction']
    )
    submission.to_csv('submission.csv', index=False)


























