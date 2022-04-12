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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, hp):
        super(BertEmbeddings, self).__init__()
        self.hp = hp
        self.word_embeddings = nn.Embedding(hp.vocab_size, hp.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(hp.max_position_embeddings, hp.hidden_size)
        if hp.do_random_next:
            self.token_type_embeddings = nn.Embedding(hp.type_vocab_size, hp.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)

    # input_ids: [bts,seq_len]
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # [seq_len]
            # bug: position_ids = position_ids.unsqueeze(0).expand_as(input_ids)，导致和hugging face结果不一样  # [bts,seq_len]
            position_ids = position_ids.unsqueeze(0)
        if self.hp.do_random_next and token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # [bts,seq_len]

        words_embeddings = self.word_embeddings(input_ids)  # [bts,seq_len,hdsz]

        #embeddings = words_embeddings


        if self.hp.do_random_next:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)  # [bts,seq_len,hdsz]
            #embeddings = embeddings + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)  # [bts,seq_len,hdsz]
        #embeddings = embeddings + position_embeddings

        if self.hp.do_random_next:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)  # [bts,seq_len,hdsz]
        embeddings = self.dropout(embeddings)  # [bts,seq_len,hdsz]
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, hp):
        super(BertSelfAttention, self).__init__()
        if hp.hidden_size % hp.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hp.hidden_size, hp.num_attention_heads))
        self.output_attentions = hp.output_attentions

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
    def forward(self, hidden_states, attention_mask, head_mask=None):
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

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # [bts,head_num,seq_len,head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bts,seq_len,head_num,head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # [bts,seq_len,hdsz]

        # ([bts,seq_len,hdsz],)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


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
    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)  # ([bts,seq_len,hdsz],)
        attention_output = self.output(self_outputs[0], input_tensor)  # [bts, seq_len, hdsz]
        # ([bts,seq_len,hdsz],)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, hp):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.intermediate_size)
        if isinstance(hp.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(hp.hidden_act, unicode)):
            self.intermediate_act_fn = nn.functional.gelu
        else:
            self.intermediate_act_fn = hp.hidden_act

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
        attention_outputs = self.attention(hidden_states, attention_mask, None)  # ([bts,seq_len,hdsz],)
        attention_output = attention_outputs[0]  # [bts,seq_len,hdsz]
        intermediate_output = self.intermediate(attention_output)  # [bts,seq_len,mdsz]
        layer_output = self.output(intermediate_output, attention_output)  # [bts,seq_len,hdsz]
        if self.hp.use_checkpoint_sequential:
            outputs = (layer_output, attention_mask.detach())
        else:
            # ([bts,seq_len,hdsz],)
            outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


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
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)  # ([bts,seq_len,hdsz],)
            hidden_states = layer_outputs[0]  # [bts,seq_len,hdsz]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # ([bts,seq_len,hdsz],)
        return outputs  # outputs, (hidden states), (attentions)


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
        self.pooler = BertPooler(hp)

    # input_ids: [bts,seq_len]
    # token_type_ids: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.hp.do_random_next and token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # [bts,seq_len]

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
        #extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # [bts,1,1,seq_len]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # [bts,1,1,seq_len]


        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.hp.num_hidden_layers  # [num_layer]

        # [bts,seq_len,hdsz]
        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        if self.hp.use_checkpoint_sequential:
            # ([bts,seq_len,hdsz], [bts,1,1,seq_len])
            encoder_outputs = checkpoint_sequential(self.encoder.layer, self.hp.chunks,
                                                    embedding_output, extended_attention_mask)
        else:
            # ([bts,seq_len,hdsz],)
            encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoder_outputs[0]  # [bts,seq_len,hdsz]
        pooled_output = self.pooler(sequence_output)  # [bts,hdsz]

        # add hidden_states and attentions if they are here
        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hp):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hp.hidden_size, hp.hidden_size)
        if isinstance(hp.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(hp.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[hp.hidden_act]
        else:
            self.transform_act_fn = hp.hidden_act
        self.LayerNorm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)

    # hidden_states: [bts,seq_len,hdsz]
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.transform_act_fn(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.LayerNorm(hidden_states)  # [bts,seq_len,hdsz]
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hp):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hp)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hp.hidden_size, hp.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hp.vocab_size))

    # hidden_states: [bts,seq_len,hdsz]
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # [bts,seq_len,hdsz]
        hidden_states = self.decoder(hidden_states) + self.bias  # [bts,seq_len,vocab_size]
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, hp):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(hp)
        self.seq_relationship = nn.Linear(hp.hidden_size, 2)

    # sequence_output: [bts,seq_len,hdsz]
    # pooled_output: [bts,hdsz]
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)  # [bts,seq_len,vocab_size]
        seq_relationship_score = self.seq_relationship(pooled_output)  # [bts,2]
        return prediction_scores, seq_relationship_score


class BertPreTrainingOneSegmentHeads(nn.Module):
    def __init__(self, hp):
        super(BertPreTrainingOneSegmentHeads, self).__init__()
        self.predictions = BertLMPredictionHead(hp)

    # sequence_output: [bts,seq_len,hdsz]
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)  # [bts,seq_len,vocab_size]
        return prediction_scores


class BertForPreTraining(nn.Module):
    def __init__(self, hp):
        super(BertForPreTraining, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        self.cls = BertPreTrainingHeads(hp)

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    # input_ids: [bts,seq_len]
    # token_type_ids: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    # masked_lm_labels: [bts,seq_len]
    # next_sentence_label: [bts]
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, position_ids=None, head_mask=None):
        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # sequence_output: [bts,seq_len,hdsz]
        # pooled_output: [bts,hdsz]
        sequence_output, pooled_output = outputs[:2]
        # prediction_scores: [bts,seq_len,vocab_size]
        # seq_relationship_score: [bts,2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        # add hidden states and attention if they are here
        # ([bts,seq_len,vocab_size],[bts,2],)
        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.hp.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class BertForPreTrainingOneSegment(nn.Module):
    def __init__(self, hp):
        super(BertForPreTrainingOneSegment, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        self.cls = BertPreTrainingOneSegmentHeads(hp)

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    # input_ids: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    # masked_lm_labels: [bts,seq_len]
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, position_ids=None, head_mask=None):
        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]  # [bts,seq_len,hdsz]
        prediction_scores = self.cls(sequence_output)  # [bts,seq_len,vocab_size]

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.hp.vocab_size), masked_lm_labels.view(-1))
        if self.hp.output_logit:
            return masked_lm_loss, prediction_scores
        else:
            return masked_lm_loss


class BertForPreTrainingSOP(nn.Module):
    def __init__(self, hp):
        super(BertForPreTrainingSOP, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        self.cls = BertPreTrainingHeads(hp)

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    # input_ids: [bts,seq_len]
    # token_type_ids: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    # masked_lm_labels: [bts,seq_len]
    # sentence_order_label: [bts]
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                sentence_order_label=None, position_ids=None, head_mask=None):
        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # sequence_output: [bts,seq_len,hdsz]
        # pooled_output: [bts,hdsz]
        sequence_output, pooled_output = outputs[:2]
        # prediction_scores: [bts,seq_len,vocab_size]
        # seq_relationship_score: [bts,2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.hp.vocab_size), masked_lm_labels.view(-1))
        sentence_order_loss = loss_fct(seq_relationship_score.view(-1, 2), sentence_order_label.view(-1))
        total_loss = masked_lm_loss + sentence_order_loss
        if self.hp.output_logit:
            return total_loss, prediction_scores, seq_relationship_score
        else:
            return total_loss





class SimcsePooler(nn.Module):
    def __init__(self, hp):
        super(SimcsePooler, self).__init__()
        self.hp = hp
        if self.hp.pooler_type == "cls":
            self.dense = nn.Linear(self.hp.hidden_size, self.hp.hidden_size)
            self.fn = nn.Tanh()

    def forward(self, output, attention_mask):
        if self.hp.pooler_type == "cls":
            pooler_vector = output[0][:,0,:]
            pooler_vector = self.dense(pooler_vector)
            pooler_vector = self.fn(pooler_vector)
        elif self.hp.pooler_type == "cls_before_pooler":
            pooler_vector = output[0][:, 0, :]
        elif self.hp.pooler_type == "avg":
            pooler_vector = torch.sum(output[0] * attention_mask.unsqueeze(dim=2),dim=1) / ( torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)
        elif self.hp.pooler_type == "avg_top2":
            pooler_vector_a = torch.sum(output[2][-1] * attention_mask.unsqueeze(dim=2),
                                      dim=1) / ( torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8)
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


class BertForSimCSE(nn.Module):
    def __init__(self, hp):
        super(BertForSimCSE, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        self.simcse_pool = SimcsePooler(hp)
        self.apply(self.init_weights)
        if self.hp.model_name == "esimcse":
            self.sigma = torch.nn.Parameter(data=torch.Tensor([self.hp.sigma]), requires_grad=True)
        

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                sentence_order_label=None,
                position_ids=None, head_mask=None, triple=None, neg_sample=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        pooler_output = self.simcse_pool(outputs, attention_mask)
        bts = pooler_output.shape[0]
        if triple:
            pair_num = bts // 3
        else:
            pair_num = bts // 2

        output_a = pooler_output[0:pair_num]
        output_b = pooler_output[pair_num:pair_num*2]
        output_a = output_a / torch.norm(output_a, dim=1,keepdim=True)
        output_b = output_b / torch.norm(output_b, dim=1, keepdim=True)
        cosine = torch.sum(output_a.unsqueeze(dim=1) * output_b.unsqueeze(dim=0), dim=-1) / self.hp.temp
        if triple:
            output_c = pooler_output[pair_num*2:]
            output_c = output_c / torch.norm(output_c, dim=1, keepdim=True)
            cosine = torch.cat([cosine,
                                torch.sum(output_a.unsqueeze(dim=1) * output_c.unsqueeze(dim=0) , dim=-1) / self.hp.temp],
                               dim=1)

        # 构建新的momentum
        if self.hp.model_name == "esimcse" and neg_sample != None:
            neg_sample = neg_sample / torch.norm(neg_sample, dim=1, keepdim=True)
            cosine = torch.cat([cosine,torch.sum(output_a.unsqueeze(dim=1) * neg_sample.unsqueeze(dim=0), dim=-1) / self.hp.temp],dim=1)

        # 自己构建label
        labels = torch.arange(cosine.size(0)).long().to(output_b.device)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss_cosine = loss_fct(cosine, labels)
        return loss_cosine


    def infer_similarity(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        temp = self.hp.pooler_type
        # 测试所有pooler_type的效果
        return_dict = {}
        if self.hp.pooler_type == "cls":
            pooler_output = self.simcse_pool(outputs,attention_mask)
            return_dict["cls"] = self._similary(pooler_output)


        self.hp.pooler_type = "cls_before_pooler"
        pooler_output = self.simcse_pool(outputs,attention_mask)
        return_dict["cls_before_pooler"] = self._similary(pooler_output)

        self.hp.pooler_type = "avg"
        pooler_output = self.simcse_pool(outputs,attention_mask)
        return_dict["avg"] = self._similary(pooler_output)


        self.hp.pooler_type = "avg_top2"
        pooler_output = self.simcse_pool(outputs,attention_mask)
        return_dict["avg_top2"] =self._similary(pooler_output)

        self.hp.pooler_type = "avg_first_last"
        pooler_output = self.simcse_pool(outputs,attention_mask)
        return_dict["avg_first_last"] = self._similary(pooler_output)

        self.hp.pooler_type = temp
        return return_dict


    def _similary(self, pooler_output):
        pooler_output_a = pooler_output[0: pooler_output.shape[0] // 2]
        pooler_output_a_norm = pooler_output_a / torch.norm(pooler_output_a, dim=1, keepdim=True)
        pooler_output_b = pooler_output[pooler_output.shape[0] // 2:]
        pooler_output_b_norm = pooler_output_b / torch.norm(pooler_output_b, dim=1, keepdim=True)
        return torch.sum(pooler_output_a_norm * pooler_output_b_norm, dim=1)

    def infer_hidden_state(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                           triple=False):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        if self.hp.pooler_type == "cls":
            pooler_output = self.simcse_pool(outputs, attention_mask)

        elif self.hp.pooler_type == "cls_before_pooler":
            pooler_output = self.simcse_pool(outputs, attention_mask)


        elif self.hp.pooler_type == "avg":
            pooler_output = self.simcse_pool(outputs, attention_mask)


        elif self.hp.pooler_type == "avg_top2":
            pooler_output = self.simcse_pool(outputs, attention_mask)


        elif self.hp.pooler_type == "avg_first_last":
            pooler_output = self.simcse_pool(outputs, attention_mask)
        else:
            raise ValueError("not suppoet `{self.hp.pooler_type}` pooler_type")

        if triple:
            pooler_output = pooler_output[0: pooler_output.shape[0] // 3]
        else:
            pooler_output = pooler_output[0:  pooler_output.shape[0] // 2]
        return pooler_output


class BertForNER(nn.Module):
    def __init__(self, hp):
        super(BertForNER, self).__init__()
        self.hp = hp
        self.bert = BertModel(hp)
        if hp.use_lstm:
            self.lstm = torch.nn.LSTM(hp.hidden_size, hp.hidden_size, num_layers=hp.lstm_layers_num,
            bidirectional=hp.lstm_bidirectional, batch_first=True)
        if hp.use_lstm and hp.lstm_bidirectional:
            feature_size = hp.hidden_size * 2
        else:
            feature_size = hp.hidden_size
        self.fc = nn.Linear(feature_size, hp.num_labels + 2)
        self.crf = CRF(hp.num_labels)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hp.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # input_ids: [bts,seq_len]
    # labels: [bts,seq_len]
    # attention_mask: [bts,seq_len]
    def forward(self, input_ids, labels, seqs_length, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        # ([bts,seq_len,hdsz],[bts,hdsz])
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]  # [bts, seq_len, hdsz]
        if self.hp.use_lstm:
            sequence_output_packed = pack_padded_sequence(sequence_output, seqs_length, batch_first=True)
            sequence_output_packed, (h_n, c_n) = self.lstm(sequence_output_packed)
            sequence_output, _ = pad_packed_sequence(sequence_output_packed, batch_first=True)
        emission_score = self.fc(sequence_output)  # [bts, seq_len, num_labels + 2]
        loss = self.crf.neg_log_likelihood_loss(emission_score, attention_mask, labels)
        return loss

    # feats: [bts, seq_len, num_labels + 2]
    # mask: [bts,seq_len]
    def predict(self, input_ids, labels=None, seqs_length=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        if self.hp.use_lstm:
            sequence_output_packed = pack_padded_sequence(sequence_output, seqs_length, batch_first=True)
            sequence_output_packed, (h_n, c_n) = self.lstm(sequence_output_packed)
            sequence_output, _ = pad_packed_sequence(sequence_output_packed, batch_first=True)
        emission_score = self.fc(sequence_output)
        loss = None
        if isinstance(labels, torch.Tensor):
            loss = self.crf.neg_log_likelihood_loss(emission_score, attention_mask, labels)
        #best_path = self.crf(emission_score, attention_mask)
        emission_score[:,:,-2:] = -1e8
        best_path = torch.argmax(emission_score,dim=2)
        return loss, best_path
