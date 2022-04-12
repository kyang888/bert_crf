# coding=utf-8
import json
import logging
import math
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from model.checkpoint import checkpoint_sequential
#from model.crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class CRF(nn.Module):
    def __init__(self, num_labels):
        super(CRF, self).__init__()
        self.num_labels = num_labels
        self.START_TAG_IDX = -2
        self.END_TAG_IDX = -1
        init_transitions = torch.zeros(self.num_labels + 2, self.num_labels + 2)
        init_transitions[:, self.START_TAG_IDX] = -10000.0
        init_transitions[self.END_TAG_IDX, :] = -10000.0
        self.transitions = nn.Parameter(init_transitions)


    def _viterbi_decode(self, feats, mask):
        bts, seq_len, tag_size = feats.size()
        ins_num = seq_len * bts
        """ calculate sentence length for each sentence """
        length_mask = torch.sum(mask, dim=1, keepdim=True).long()  # [bts, 1]
        mask = mask.transpose(1, 0).contiguous()  # [seq_len, bts]

        # [seq_len * bts, tag_size, tag_size]
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        # [seq_len * bts, tag_size, tag_size]
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, bts, tag_size, tag_size)  # [seq_len, bts, tag_size, tag_size]

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = []
        partition_history = []
        mask = (1 - mask.long()).bool()  # [seq_len, bts]
        _, inivalues = next(seq_iter)  # [bts, tag_size, tag_size]
        """ only need start from start_tag """
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(bts, tag_size, 1)  # [bts, tag_size,1]
        partition_history.append(partition)

        for idx, cur_values in seq_iter:  # scalar, [bts, tag_size, tag_size]
            # [bts, tag_size, tag_size]
            cur_values = cur_values + partition.contiguous().view(bts, tag_size, 1).expand(bts, tag_size, tag_size)
            """ do not consider START_TAG/STOP_TAG """
            partition, cur_bp = torch.max(cur_values, 1)  # [bts, tag_size], [bts, tag_size]
            partition_history.append(partition.unsqueeze(-1))
            """ set padded label as 0, which will be filtered in post processing"""
            cur_bp.masked_fill_(mask[idx].view(bts, 1).expand(bts, tag_size), 0)  # [bts, tag_size]
            back_points.append(cur_bp)

        # [bts, seq_len, tag_size]
        partition_history = torch.cat(partition_history).view(seq_len, bts, -1).transpose(1, 0).contiguous()
        """ get the last position for each setences, and select the last partitions using gather() """
        last_position = length_mask.view(bts, 1, 1).expand(bts, 1, tag_size) - 1  # [bts, 1, tag_size]
        # [bts, tag_size, 1]
        last_partition = torch.gather(partition_history, 1, last_position).view(bts, tag_size, 1)
        """ calculate the score from last partition to end state (and then select the STOP_TAG from it) """
        # [bts, tag_size, tag_size]
        last_values = last_partition.expand(bts, tag_size, tag_size) + \
                      self.transitions.view(1, tag_size, tag_size).expand(bts, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)  # [bts, tag_size]
        """ select end ids in STOP_TAG """
        pointer = last_bp[:, self.END_TAG_IDX]  # [bts]

        pad_zero = torch.zeros(bts, tag_size, requires_grad=True).to(mask).long()  # [bts, tag_size]
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, bts, tag_size)  # [seq_len, bts, tag_size]

        insert_last = pointer.contiguous().view(bts, 1, 1).expand(bts, 1, tag_size)  # [bts, 1, tag_size]
        back_points = back_points.transpose(1, 0).contiguous()  # [bts, seq_len, tag_size]
        """move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values """
        back_points.scatter_(1, last_position, insert_last)  # [bts, seq_len, tag_size]

        back_points = back_points.transpose(1, 0).contiguous()  # [seq_len, bts, tag_size]
        """ decode from the end, padded position ids are 0, which will be filtered if following evaluation """
        decode_idx = torch.empty(seq_len, bts, requires_grad=True).to(pointer)  # [seq_len, bts]
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bts, 1))
            decode_idx[idx] = pointer.view(-1).data
        decode_idx = decode_idx.transpose(1, 0)  # [bts, seq_len]
        return decode_idx

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    def forward(self, feats, mask):
        best_path = self._viterbi_decode(feats, mask)  # [bts, seq_len]
        return best_path


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, hp):
        super(BertEmbeddings, self).__init__()
        self.hp = hp
        self.word_embeddings = nn.Embedding(hp.vocab_size, hp.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(hp.max_position_embeddings, hp.hidden_size)
        self.token_type_embeddings = nn.Embedding(hp.type_vocab_size, hp.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # input_ids: [bts,seq_len]
    def forward(self, input_ids, token_type_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)  # [bts,seq_len,hdsz]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # [bts,seq_len,hdsz]
        position_embeddings = self.position_embeddings(position_ids)  # [bts,seq_len,hdsz]

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)  # [bts,seq_len,hdsz]
        embeddings = self.dropout(embeddings)  # [bts,seq_len,hdsz]
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, hp):
        super(BertSelfAttention, self).__init__()
        # if hp.hidden_size % hp.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (hp.hidden_size, hp.num_attention_heads))
        # self.output_attentions = hp.output_attentions
        #
        self.num_attention_heads = hp.num_attention_heads
        self.attention_head_size = int(hp.hidden_size / hp.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hp.hidden_size, self.all_head_size)
        self.key = nn.Linear(hp.hidden_size, self.all_head_size)
        self.value = nn.Linear(hp.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(hp.attention_probs_dropout_prob)

    # x: [bts,seq_len,hdsz]
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [bts,seq_len,head_num,head_size]
        return x.permute(0, 2, 1, 3)  # [bts,head_num,seq_len,head_size]

    # hidden_states: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: None
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  # [bts,seq_len,hdsz]
        mixed_key_layer = self.key(hidden_states)  # [bts,seq_len,hdsz]
        mixed_value_layer = self.value(hidden_states)  # [bts,seq_len,hdsz]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bts,head_num,seq_len,head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [bts,head_num,seq_len,head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bts,head_num,seq_len,head_size]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bts,head_num,seq_len,seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bts,head_num,seq_len,seq_len]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # [bts,head_num,seq_len,seq_len]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bts,head_num,seq_len,seq_len]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # [bts,head_num,seq_len,seq_len]


        context_layer = torch.matmul(attention_probs, value_layer)  # [bts,head_num,seq_len,head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bts,seq_len,head_num,head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [bts,seq_len,hdsz]

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hp):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.LayerNorm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # hidden_states: [bts, seq_len, hdsz]
    # input_tensor: [bts, seq_len, hdsz]
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # [bts, seq_len, hdsz]
        hidden_states = self.dropout(hidden_states)  # [bts, seq_len, hdsz]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # [bts, seq_len, hdsz]
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, hp):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hp)
        self.output = BertSelfOutput(hp)

    # input_tensor: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: None
    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)  # ([bts,seq_len,hdsz],)
        attention_output = self.output(self_outputs, input_tensor)  # [bts, seq_len, hdsz]
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, hp):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.intermediate_size)
        self.intermediate_act_fn = ACT2FN[hp.hidden_act]


    # hidden_states: [bts,seq_len,hdsz]
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # [bts,seq_len,mdsz]
        hidden_states = self.intermediate_act_fn(hidden_states)  # [bts,seq_len,mdsz]
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hp):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hp.intermediate_size, hp.hidden_size)
        self.LayerNorm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # hidden_states: [bts,seq_len,mdsz]
    # input_tensor: [bts,seq_len,hdsz]
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.dropout(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # [bts,seq_len,hdsz]
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hp):
        super(BertLayer, self).__init__()
        self.hp = hp
        self.attention = BertAttention(hp)
        self.intermediate = BertIntermediate(hp)
        self.output = BertOutput(hp)

    # hidden_states: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: None
    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask)  # ([bts,seq_len,hdsz],)
        intermediate_output = self.intermediate(attention_outputs)  # [bts,seq_len,mdsz]
        layer_output = self.output(intermediate_output, attention_outputs)  # [bts,seq_len,hdsz]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, hp):
        super(BertEncoder, self).__init__()
        self.output_attentions = hp.output_attentions
        self.output_hidden_states = hp.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(hp) for _ in range(hp.num_hidden_layers)])

    # hidden_states: [bts,seq_len,hdsz]
    # attention_mask: [bts,1,1,seq_len]
    # head_mask: [num_layer]
    def forward(self, hidden_states, attention_mask, head_mask=None):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)  # ([bts,seq_len,hdsz],)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, hp):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.hidden_size)
        self.activation = nn.Tanh()

    # hidden_states: [bts,seq_len,hdsz]
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]  # [bts,hdsz]
        pooled_output = self.dense(first_token_tensor)  # [bts,hdsz]
        pooled_output = self.activation(pooled_output)  # [bts,hdsz]
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, hp):
        super(BertModel, self).__init__()
        self.hp = hp
        self.embeddings = BertEmbeddings(hp)
        self.encoder = BertEncoder(hp)
        #self.pooler = BertPooler(hp)

    # input_ids: [bts,seq_len]
    # token_type_ids: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    def forward(self, input_ids,  attention_mask,token_type_ids, position_ids):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bts,1,1,seq_len]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # [bts,1,1,seq_len]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]


        # [bts,seq_len,hdsz]
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        sequence_output = self.encoder(embedding_output, extended_attention_mask)
        return sequence_output


class BertForNER(nn.Module):
    def __init__(self, hp):
        super(BertForNER, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        self.crf = CRF(hp.num_labels)
        self.fc = nn.Linear(hp.hidden_size, hp.num_labels + 2)




    def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        sequence_output = self.bert(input_ids, attention_mask, token_type_ids, position_ids)
        emission_score = self.fc(sequence_output)
        best_path = self.crf(emission_score, attention_mask)
        return best_path
