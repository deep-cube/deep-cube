import torch
import data_def
import ctcdecode
import torch.nn.functional as F


def cross_entropy_loss(scores, y):
    ce_loss = torch.nn.CrossEntropyLoss(
        reduction='mean'
    )

    return ce_loss(scores, y)


def cross_entropy_loss_mean(scores, y, loss_weights):
    ce_loss = torch.nn.CrossEntropyLoss(
        reduction='mean',
        weight=loss_weights
    )

    y_reshaped = y.reshape(-1)  # (B * L)
    scores_reshaped = scores.reshape(-1, scores.shape[-1])
    # (B * L, num_class)
    loss = ce_loss(scores_reshaped, y_reshaped)
    return loss


def ctc_loss_sum(scores, y_collapsed, y_collapsed_l):
    B, L, num_class = scores.shape
    score_log_softmax = scores.log_softmax(dim=-1).transpose(0, 1)
    input_lengths = torch.full(
        size=(B,), fill_value=L, dtype=torch.long)
    target_lengths = y_collapsed_l
    ctcloss = torch.nn.CTCLoss(
        blank=0, reduction='sum'
    )(score_log_softmax, y_collapsed, input_lengths, target_lengths)
    return ctcloss


def sum_collapsed_edit_distance_hardmax(scores, y):
    # scores (B, L, num_class)
    # y (B, L, )
    B, L = y.shape
    yhat = torch.argmax(scores, dim=-1).cpu().detach().numpy()  # (B, L,)
    y = y.cpu().numpy()  # (B, L,)
    return sum(
        data_def.edit_distance(
            data_def.ctc_collapse_to_list(yhat[i]),
            data_def.remove_space(y[i])
        )
        for i in range(B)
    )


def beam_search(scores, beam_size=50):
    B, L, _ = scores.shape
    decoder = ctcdecode.CTCBeamDecoder(
        data_def.CLASS_STRS, beam_width=beam_size, blank_id=0)
    scores_softmax = F.softmax(scores, dim=2)
    beam_result, _, _, seq_len = decoder.decode(scores_softmax)

    return beam_result, seq_len


def best_beam(scores, beam_size=50):
    B, L, _ = scores.shape
    assert B == 1
    beam_result, seq_len = beam_search(scores, beam_size)
    return beam_result[0][0][:seq_len[0][0]]


def sum_collapsed_edit_distance_beam(scores, y, beam_size=50):
    B, L = y.shape
    # decoder = ctcdecode.CTCBeamDecoder(
    #     data_def.CLASS_STRS, beam_width=beam_size, blank_id=0)
    # scores_softmax = F.softmax(scores, dim=2)
    # beam_result, _, _, seq_len = decoder.decode(scores_softmax)
    # print(beam_result.shape)
    # raise
    beam_result, seq_len = beam_search(scores, beam_size)

    return sum(
        data_def.edit_distance(
            beam_result[i][0][:seq_len[i][0]].cpu().detach().numpy(),
            data_def.remove_space(y[i].cpu().numpy())
        )
        for i in range(B)
    )


def sum_collapsed_edit_distance_trimmed(scores, y, trim_length):
    # scores (B, L, num_class)
    # y (B, L, )
    return sum_collapsed_edit_distance_hardmax(
        scores[:, trim_length:, :],
        y[:, trim_length:],
    )
