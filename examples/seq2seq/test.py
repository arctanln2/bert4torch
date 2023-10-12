import json
import os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import AutoRegressiveDecoder
import torch
import time

t2 = time.time()
# 基本参数
maxlen = 256  # 读取文本字数最大值
batch_size = 2
epochs = 10
steps_per_epoch = None

# bert配置
config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    add_trainer=True
).to(device)
model.load_weights('./bert_unilm.pt')


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat(
            [segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = model.predict([token_ids, segment_ids])
        return y_pred[:, -1, :]

    def generate(self, text, topk=1, topp=0.95):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        # print(type(token_dict),len(token_dict))# ().device,segment_ids.device)
        output_ids = self.beam_search(
            [token_ids, segment_ids], topk=topk)  # 基于beam search
        # print(output_ids.device)
        return tokenizer.decode(output_ids.cpu().numpy())


autotitle = AutoTitle(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=32, device=device)


def just_show():
    s1 = u'抽象了一种基于中心的战术应用场景与业务,并将网络编码技术应用于此类场景的实时数据多播业务中。在分析基于中心网络与Many-to-all业务模式特性的基础上,提出了仅在中心节点进行编码操作的传输策略以及相应的贪心算法。分析了网络编码多播策略的理论增益上界,仿真试验表明该贪心算法能够获得与理论相近的性能增益。最后的分析与仿真试验表明,在这种有中心网络的实时数据多播应用中,所提出的多播策略的实时性能要明显优于传统传输策略。'
    s2 = u'普适计算环境中未知移动节点的位置信息是定位服务要解决的关键技术。在普适计算二维空间定位过程中,通过对三角形定位单元区域的误差分析,提出了定位单元布局(LUD)定理。在此基础上,对多个定位单元布局进行了研究,定义了一个新的描述定位单元中定位参考点覆盖效能的物理量——覆盖基,提出了在误差最小情况下定位单元布局的覆盖基定理。仿真实验表明定位单元布局定理能更好地满足对普适终端实时定位的需求,且具有较高的精度和最大覆盖效能。'
    print(u'生成标题:', autotitle.generate(s2))


t1 = time.time()
print(f"init cost {t1-t2}s")
just_show()
t2 = time.time()
print(f"predict cost {t2-t1}s")
