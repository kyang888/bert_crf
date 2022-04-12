


# coding=utf-8
import json
import logging
import math
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from model.checkpoint import checkpoint_sequential
from model.crf import CRF

logger = logging.getLogger(__name__)


from transformers import BertModel,BertPreTrainedModel


class SimcsePooler(nn.Module):
    def __init__(self, hp):
        super(SimcsePooler, self).__init__()
        self.hp = hp
        if self.hp.pooler_type == "cls":
            self.dense = nn.Linear(self.hp.hidden_size, self.hp.hidden_size)
            self.fn = nn.Tanh()

    def forward(self, output, attention_mask):
        if self.hp.pooler_type == "cls":
            pooler_vector = output[0][:, 0, :]
            pooler_vector = self.dense(pooler_vector)
            pooler_vector = self.fn(pooler_vector)
        elif self.hp.pooler_type == "cls_before_pooler":
            pooler_vector = output[0][:, 0, :]
        elif self.hp.pooler_type == "avg":
            pooler_vector = torch.sum(output[0] * attention_mask.unsqueeze(dim=2), dim=1) / (
                        torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)
        elif self.hp.pooler_type == "avg_top2":
            pooler_vector_a = torch.sum(output[2][-1] * attention_mask.unsqueeze(dim=2),
                                        dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)
            pooler_vector_b = torch.sum(output[2][-2] * attention_mask.unsqueeze(dim=2),
                                        dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)

            pooler_vector = (pooler_vector_a + pooler_vector_b) / 2.0
        elif self.hp.pooler_type == "avg_first_last":
            pooler_vector_a = torch.sum(output[2][-1] * attention_mask.unsqueeze(dim=2),
                                        dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)
            pooler_vector_b = torch.sum(output[2][1] * attention_mask.unsqueeze(dim=2),
                                        dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)

            pooler_vector = (pooler_vector_a + pooler_vector_b) / 2.0
        return pooler_vector


# class BertQueryNER(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertQueryNER, self).__init__(config)
#         self.bert = BertModel(config)


class BertForSimCSE(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSimCSE, self).__init__(config)
        self.bert = BertModel(config)

        self.init_weights()

    def AAA(self, hp):
        self.hp = hp
        self.simcse_pool = SimcsePooler(hp)


    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                sentence_order_label=None,
                position_ids=None, head_mask=None, triple=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, output_hidden_states=self.hp.output_hidden_states)

        pooler_output = self.simcse_pool(outputs, attention_mask)
        bts = pooler_output.shape[0]
        if triple:
            pair_num = bts // 3
        else:
            pair_num = bts // 2

        output_a = pooler_output[0:pair_num]
        output_b = pooler_output[pair_num:pair_num * 2]
        output_a = output_a / torch.norm(output_a, dim=1, keepdim=True)
        output_b = output_b / torch.norm(output_b, dim=1, keepdim=True)
        if triple:
            output_c = pooler_output[pair_num * 2:]
            output_c = output_c / torch.norm(output_c, dim=1, keepdim=True)

        cosine = torch.sum(output_a.unsqueeze(dim=1) * output_b.unsqueeze(dim=0), dim=-1) / self.hp.temp
        if triple:
            cosine = torch.cat(cosine,
                               torch.sum(output_a.unsqueeze(dim=1) * output_c.unsqueeze(dim=0), dim=-1) / self.hp.temp)

        # 自己构建label
        labels = torch.arange(cosine.size(0)).long().to(output_b.device)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss_cosine = loss_fct(cosine, labels)
        return loss_cosine

    def infer_hidden_state(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                           head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask,output_hidden_states=self.hp.output_hidden_states)
        temp = self.hp.pooler_type
        # 测试所有pooler_type的效果
        return_dict = {}
        if self.hp.pooler_type == "cls":
            pooler_output = self.simcse_pool(outputs, attention_mask)
            return_dict["cls"] = self._similary(pooler_output)

        self.hp.pooler_type = "cls_before_pooler"
        pooler_output = self.simcse_pool(outputs, attention_mask)
        return_dict["cls_before_pooler"] = self._similary(pooler_output)

        self.hp.pooler_type = "avg"
        pooler_output = self.simcse_pool(outputs, attention_mask)
        return_dict["avg"] = self._similary(pooler_output)

        self.hp.pooler_type = "avg_top2"
        pooler_output = self.simcse_pool(outputs, attention_mask)
        return_dict["avg_top2"] = self._similary(pooler_output)

        self.hp.pooler_type = "avg_first_last"
        pooler_output = self.simcse_pool(outputs, attention_mask)
        return_dict["avg_first_last"] = self._similary(pooler_output)

        self.hp.pooler_type = temp
        return return_dict

    def _similary(self, pooler_output):
        pooler_output_a = pooler_output[0: pooler_output.shape[0] // 2]
        pooler_output_a_norm = pooler_output_a / torch.norm(pooler_output_a, dim=1, keepdim=True)
        pooler_output_b = pooler_output[pooler_output.shape[0] // 2:]
        pooler_output_b_norm = pooler_output_b / torch.norm(pooler_output_b, dim=1, keepdim=True)
        return torch.sum(pooler_output_a_norm * pooler_output_b_norm, dim=1)


class BertForSB(nn.Module):
    def __init__(self, hp):
        super(BertForSB, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        self.classifier = nn.Linear(hp.hidden_size * 3, hp.num_labels)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # input_ids: [bts,seq_len]
    # labels: [bts]
    # attention_mask: [bts,seq_len]
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label=None, sentence_order_label=None,
                position_ids=None, head_mask=None):
        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]  # [bts, len, hdsz]
        # 取第一部分
        bts = input_ids.shape[0]

        # 取cls位向量
        sequence_output_mean = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask.float(), dim=1,
                                                                             keepdim=True)
        # sequence_output = sequence_output[:, 0, :]
        output_a = sequence_output_mean[0:bts // 2]
        output_b = sequence_output_mean[bts // 2:]

        # output_a_norm = output_a / torch.norm(output_a, dim=1, keepdim=True)
        # output_b_norm = output_b / torch.norm(output_b, dim=1, keepdim=True)

        output_concat = torch.cat((output_a, output_b, torch.abs(output_a - output_b)), dim=1)

        output_norm = output_concat / torch.norm(output_concat, dim=1, keepdim=True)

        logit = self.classifier(output_norm)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logit, label)
        return loss

    def infer_hidden_state(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                           head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output_mean = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask.float(), dim=1,
                                                                             keepdim=True)
        sequence_output_norm = sequence_output_mean / torch.norm(sequence_output_mean, dim=1, keepdim=True)
        return sequence_output_norm

    def computer_cosine(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        """
        取最后一层cls位进行初始化
        """
        bts = input_ids.shape[0]
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]  # [bts, len, hdsz]
        # sum_output = torch.sum(    sequence_output *  attention_mask.unsqueeze(dim=2).to(sequence_output.dtype),          dim=1   )
        # mean_output = sum_output / torch.sum(attention_mask, dim=1, keepdim=True).to(sequence_output.dtype)
        mean_output = sequence_output[:, 0, :]
        mean_output_a = mean_output[0:bts // 2]
        mean_output_b = mean_output[bts // 2:]
        mean_output_a = mean_output_a / torch.norm(mean_output_a, dim=1, keepdim=True)
        mean_output_b = mean_output_b / torch.norm(mean_output_b, dim=1, keepdim=True)
        return torch.sum(mean_output_a * mean_output_b, dim=1)

