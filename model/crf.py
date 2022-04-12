import torch
import torch.nn as nn


# vec: [bts, tag_size, tag_size]
# tag_size: scalar
# def log_sum_exp(vec, tag_size):
#     max_score, _ = torch.max(vec, dim=1, keepdim=True)  # [bts, 1, tag_size]
#     # [bts, tag_size]
#     return max_score.view(-1, tag_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1))

def log_sum_exp(vec, m_size):
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


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

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    def _forward_alg(self, feats, mask):
        bts, seq_len, tag_size = feats.size()
        ins_num = bts * seq_len

        mask = mask.transpose(1, 0).contiguous()  # [seq_len, bsz]

        # [seq_len * bts, tag_size, tag_size]
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        # [seq_len * bts, tag_size, tag_size]
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, bts, tag_size, tag_size)  # [seq_len, bts, tag_size, tag_size]
        seq_iter = enumerate(scores)

        """ only need start from start_tag """
        _, inivalues = next(seq_iter)  # [bts, tag_size, tag_size]
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(bts, tag_size, 1)  # [bts, tag_size, 1]

        for idx, cur_values in seq_iter:  # scalar, [bts, tag_size, tag_size]
            # [bts, tag_size, tag_size]
            cur_values = cur_values + partition.contiguous().view(bts, tag_size, 1).expand(bts, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)  # [bts, tag_size]
            mask_idx = mask[idx, :].view(bts, 1).expand(bts, tag_size)  # [bts, tag_size]
            """ effective updated partition part, only keep the partition value of mask value = 1 """
            masked_cur_partition = cur_partition.masked_select(mask_idx.bool())  # [x * tag_size]
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(bts, tag_size, 1)  # [bts, tag_size, 1]
                """ replace the partition where the maskvalue=1, other partition value keeps the same """
                partition.masked_scatter_(mask_idx.bool(), masked_cur_partition)
        # [bts, tag_size, tag_size]
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(bts, tag_size, tag_size) + \
                     partition.contiguous().view(bts, tag_size, 1).expand(bts, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)  # [bts, tag_size]
        final_partition = cur_partition[:, self.END_TAG_IDX]  # [bts]
        return final_partition.sum(), scores

    # scores: [seq_len, bts, tag_size, tag_size]
    # mask: [bts, seq_len]
    # tags: [bts, seq_len]
    def _score_sentence(self, scores, mask, tags):
        seq_len, btz, tag_size, _ = scores.size()

        """ convert tag value into a new format, recorded label bigram information to index """
        new_tags = torch.empty(btz, seq_len, requires_grad=True).to(tags)  # [btz, seq_len]
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]  # `tag_size - 2` account for `START_TAG_IDX`
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, btz, 1)  # [seq_len, btz, 1]

        # get all energies except end energy
        tg_energy = torch.gather(scores.view(seq_len, btz, -1), 2, new_tags).view(seq_len, btz)  # [seq_len, btz]
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0).bool())  # list

        """ transition for label to STOP_TAG """
        # [btz, tag_size]
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(1, tag_size).expand(btz, tag_size)
        """ length for batch,  last word position = length - 1 """
        length_mask = torch.sum(mask, dim=1, keepdim=True).long()  # [bts, 1]
        """ index the label id of last word """
        end_ids = torch.gather(tags, 1, length_mask - 1)  # [bts, 1]
        """ index the transition score for end_id to STOP_TAG """
        end_energy = torch.gather(end_transition, 1, end_ids)  # [bts, 1]

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
    # tags: [bts, seq_len]
    def neg_log_likelihood_loss(self, feats, mask, tags):
        bts = feats.size(0)
        # scalar, [seq_len, bts, tag_size, tag_size]
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return (forward_score - gold_score) / bts

    # feats: [bts, seq_len, num_labels+2]
    # mask: [bts, seq_len]
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
