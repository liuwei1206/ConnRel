# author = liuwei
# date = 2022-04-11
import math
import os
import json

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaForMaskedLM


class RoBERTaForRelCls(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(RoBERTaForRelCls, self).__init__(config)

        self.roberta = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.num_labels = args.num_labels

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        flag="Train"
    ):
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooling_outputs = roberta_outputs.pooler_output
        pooling_outputs = self.dropout(pooling_outputs)
        logits = self.classifier(pooling_outputs)
        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs


class RobertaForConnCls(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(RobertaForConnCls, self).__init__(config)

        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.num_connectives = args.num_connectives
        self.pooling_type = args.pooling_type
        if self.pooling_type.lower() == "cls":
            self.conn_roberta = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
            self.classifier = nn.Linear(config.hidden_size, args.num_connectives)
        else:
            self.conn_roberta = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
            self.conn_onehot_in_vocab = args.conn_onehot_in_vocab  # [conn_num, vocab_size]
            self.conn_length_in_vocab = args.conn_length_in_vocab  # [conn_num]

    def forward(
        self,
        input_ids,
        attention_mask,
        mask_position_ids,
        conn_ids=None,
        flag="Train"
    ):
        if self.pooling_type.lower() == "cls":
            roberta_outputs = self.conn_roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooling_outputs = roberta_outputs.pooler_output
            pooling_outputs = self.dropout(pooling_outputs)
            conn_logits = self.classifier(pooling_outputs)
        else:
            # we use the <mask>
            roberta_outputs = self.conn_roberta.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            last_hidden_states = roberta_outputs.last_hidden_state  # [N, L, D]
            hidden_size = last_hidden_states.size(2)
            mask_position_index = mask_position_ids.view(-1, 1, 1)  # [N, 1, 1]
            mask_position_index = mask_position_index.repeat(1, 1, hidden_size)  # [N, 1, D]
            mask_token_states = torch.gather(last_hidden_states, dim=1, index=mask_position_index)  # [N, 1, D]
            mask_token_states = mask_token_states.squeeze()  # [N, D]

            # 1.3 make use of masked_language_linear function
            mask_token_states = self.conn_roberta.lm_head.dense(mask_token_states)
            mask_token_states = gelu(mask_token_states)
            mask_token_states = self.conn_roberta.lm_head.layer_norm(mask_token_states)  # [N, D]
            conn_decoder_weight = torch.matmul(self.conn_onehot_in_vocab, self.conn_roberta.lm_head.decoder.weight)  # [conn_num, D]
            conn_decoder_bias = torch.matmul(self.conn_onehot_in_vocab, self.conn_roberta.lm_head.decoder.bias.unsqueeze(1))  # [conn_num, 1]
            conn_decoder_weight = conn_decoder_weight / self.conn_length_in_vocab.unsqueeze(1)
            conn_decoder_bias = conn_decoder_bias / self.conn_length_in_vocab.unsqueeze(1)
            conn_decoder_weight = torch.transpose(conn_decoder_weight, 1, 0)  # [D, conn_num]
            conn_decoder_bias = torch.transpose(conn_decoder_bias, 1, 0)  # [1, conn_num]
            conn_logits = torch.matmul(mask_token_states, conn_decoder_weight) + conn_decoder_bias  # [N, conn_num]

        _, preds = torch.max(conn_logits, dim=-1)
        outputs = (preds,)
        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(conn_logits.view(-1, self.num_connectives), conn_ids.view(-1))
            outputs = (loss,) + outputs

        return outputs


class MultiTaskForConnRelCls(RobertaPreTrainedModel):
    """
    Refer to paper: Adapting BERT to Implicit Discourse Relation Classification
                    with a Focus on Discourse Connectives
    """
    def __init__(self, config, args):
        super(MultiTaskForConnRelCls, self).__init__(config)

        self.roberta = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.num_labels = args.num_labels
        self.num_connectives = args.num_connectives
        self.conn_classifier = nn.Linear(config.hidden_size, args.num_connectives)
        self.rel_classifier = nn.Linear(config.hidden_size, args.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        conn_ids=None,
        labels=None,
        flag="Train"
    ):
        roberta_outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooling_outputs = roberta_outputs.pooler_output
        pooling_outputs = self.dropout(pooling_outputs)
        conn_logits = self.conn_classifier(pooling_outputs)
        rel_logits = self.rel_classifier(pooling_outputs)
        conn_preds = torch.argmax(conn_logits, dim=-1)
        rel_preds = torch.argmax(rel_logits, dim=-1)
        outputs = (conn_preds, rel_preds, )

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            conn_loss = loss_fct(conn_logits.view(-1, self.num_connectives), conn_ids.view(-1))
            rel_loss = loss_fct(rel_logits.view(-1, self.num_labels), labels.view(-1))
            loss = conn_loss + rel_loss
            outputs = (loss, conn_loss, rel_loss) + outputs

        return outputs


class AdversarialModelForRelCls(RobertaPreTrainedModel):
    def __init__(self, config, args):
        """
        Refer to paper: Adversarial Connective-exploiting Networks for Implicit
        Discourse Relation Classification
        """
        super(AdversarialModelForRelCls, self).__init__(config)

        # roberta_ori: origin implicit relation,
        # roberta_aug: connectives argument implicit relation,
        self.roberta_ori = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
        self.roberta_arg = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.num_labels = args.num_labels

        # relation classifier and discriminator classifier
        self.rel_classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.dis_classifier = nn.Linear(config.hidden_size, 2) # contain connectives or not

        self.fix_roberta_arg = False
        self.fix_roberta_ori = False

    def set_roberta_arg(self, do_fix=True):
        if do_fix:
            self.fix_roberta_arg = True
            for name, param in self.roberta_arg.named_parameters():
                param.requires_grad = False
        else:
            self.fix_roberta_arg = False
            for name, param in self.roberta_arg.named_parameters():
                param.requires_grad = True

    def conn_arg_rel_forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        flag="Train"
    ):
        """ Train only connective argument relation classification model """
        roberta_outputs = self.roberta_arg(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooling_outputs = roberta_outputs.pooler_output
        pooling_outputs = self.dropout(pooling_outputs)
        logits = self.rel_classifier(pooling_outputs)
        preds = torch.argmax(logits, dim=-1)
        outputs = (preds, )
        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs

    def set_roberta_ori(self, do_fix=True):
        if do_fix:
            self.fix_roberta_ori = True
            for name, param in self.roberta_ori.named_parameters():
                param.requires_grad = False
        else:
            self.fix_roberta_ori = False
            for name, param in self.roberta_ori.named_parameters():
                param.requires_grad = True

    def origin_rel_forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
        flag="Train"
    ):
        """ Train original Relation classification model, no connectives in inputs """
        roberta_outputs = self.roberta_ori(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooling_outputs = roberta_outputs.pooler_output
        pooling_outputs = self.dropout(pooling_outputs)
        logits = self.rel_classifier(pooling_outputs)
        preds = torch.argmax(logits, dim=-1)
        outputs = (preds,)
        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def discriminator_forward(
        self,
        input_ids,
        attention_mask,
        arg_input_ids,
        arg_attention_mask,
        flag="Train"
    ):
        """
            In this stage, discriminators try to distinguish which output
            contains connectives, binary classification
        """
        batch_size = input_ids.size(0)
        with torch.no_grad():
            ori_roberta_outputs = self.roberta_ori(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            arg_roberta_outputs = self.roberta_arg(
                input_ids=arg_input_ids,
                attention_mask=arg_attention_mask
            )

        ori_pooling_outputs = ori_roberta_outputs.pooler_output
        arg_pooling_outputs = arg_roberta_outputs.pooler_output
        ori_pooling_outputs = self.dropout(ori_pooling_outputs)
        arg_pooling_outputs = self.dropout(arg_pooling_outputs)
        ori_logits = self.dis_classifier(ori_pooling_outputs)
        arg_logits = self.dis_classifier(arg_pooling_outputs)
        ori_preds = torch.argmax(ori_logits, dim=-1)
        arg_preds = torch.argmax(arg_logits, dim=-1)
        outputs = (ori_preds, arg_preds, )

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ori_labels = torch.zeros((batch_size)).long().to(input_ids.device)
            arg_labels = torch.ones((batch_size)).long().to(input_ids.device)
            labels = torch.cat((ori_labels, arg_labels), dim=0)
            logits = torch.cat((ori_logits, arg_logits), dim=0)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def joint_forward(
        self,
        input_ids,
        attention_mask,
        labels,
        flag="Train"
    ):
        """
            In this stage, model try to correctly predict the relation class
            and to update origin roberta's output so that discriminator predicts it
            as 1, i.e. output contains connectives.
        """
        batch_size = input_ids.size(0)
        ori_roberta_outputs = self.roberta_ori(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        ori_pooling_outputs = ori_roberta_outputs.pooler_output
        ori_pooling_outputs = self.dropout(ori_pooling_outputs)
        logits = self.rel_classifier(ori_pooling_outputs)
        dis_logits = self.dis_classifier(ori_pooling_outputs)
        preds = torch.argmax(logits, dim=-1)
        dis_preds = torch.argmax(dis_logits, dim=-1)
        outputs = (preds, dis_preds, )
        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            dis_labels = torch.ones((batch_size)).long().to(input_ids.device)
            rel_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            dis_loss = loss_fct(dis_logits.view(-1, 2), dis_labels.view(-1))
            loss = rel_loss + 0.1 * dis_loss # refer to origin paper for weight assignment
            outputs = (loss, ) + outputs

        return outputs


class JointConnRel(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(JointConnRel, self).__init__(config)

        self.conn_roberta = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.num_connectives = args.num_connectives
        self.num_labels = args.num_labels
        self.conn_onehot_in_vocab = args.conn_onehot_in_vocab  # [conn_num, vocab_size]
        self.conn_length_in_vocab = args.conn_length_in_vocab  # [conn_num]

    def forward(
            self,
            input_ids,
            attention_mask,
            mask_position_ids,
            sample_p=None,
            conn_ids=None,
            labels=None,
            flag="Train"
    ):
        """
        batch_size: N
        seq_length: L
        hidden_size: D
        Args:
            input_ids: [N, L], args1 [mask] args2
            attention_mask: [N, L], 哪些位置是有效的
            mask_position_ids: [N], the position of [mask] tokens
            conn_ids: [N], ground truth connective ids
            labels: [N], relation labels
        """
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        ## 1 for discourse connective prediction
        # 1.1 roberta
        conn_roberta_output = self.conn_roberta.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_states = conn_roberta_output.last_hidden_state  # [N, L, D]
        hidden_size = last_hidden_states.size(2)

        # 1.2 scatter the mask position
        mask_position_index = mask_position_ids.view(-1, 1, 1)  # [N, 1, 1]
        mask_position_index = mask_position_index.repeat(1, 1, hidden_size)  # [N, 1, D]
        mask_token_states = torch.gather(last_hidden_states, dim=1, index=mask_position_index)  # [N, 1, D]
        mask_token_states = mask_token_states.squeeze()  # [N, D]

        # 1.3 make use of masked_language_linear function, refer to LMHead in Roberta
        mask_token_states = self.conn_roberta.lm_head.dense(mask_token_states)
        mask_token_states = gelu(mask_token_states)
        mask_token_states = self.conn_roberta.lm_head.layer_norm(mask_token_states)  # [N, D]
        conn_decoder_weight = torch.matmul(self.conn_onehot_in_vocab, self.conn_roberta.lm_head.decoder.weight)  # [conn_num, D]
        conn_decoder_bias = torch.matmul(self.conn_onehot_in_vocab, self.conn_roberta.lm_head.decoder.bias.unsqueeze(1))  # [conn_num, 1]
        conn_decoder_weight = conn_decoder_weight / self.conn_length_in_vocab.unsqueeze(1)
        conn_decoder_bias = conn_decoder_bias / self.conn_length_in_vocab.unsqueeze(1)
        conn_decoder_weight = torch.transpose(conn_decoder_weight, 1, 0)  # [D, conn_num]
        conn_decoder_bias = torch.transpose(conn_decoder_bias, 1, 0)  # [1, conn_num]
        conn_logits = torch.matmul(mask_token_states, conn_decoder_weight) + conn_decoder_bias  # [N, conn_num]

        if self.training:
            p = random.random()
            if p < sample_p:
                conn_scores = conn_ids
                ones = torch.eye(self.num_connectives).to(conn_scores.device)
                conn_scores = ones.index_select(dim=0, index=conn_scores)
            else:
                conn_scores = F.gumbel_softmax(conn_logits, tau=1.0, hard=True, dim=-1)
        else:
            conn_scores = torch.argmax(conn_logits, dim=-1)
            # conn_scores = conn_ids
            ones = torch.eye(self.num_connectives).to(conn_scores.device)
            conn_scores = ones.index_select(dim=0, index=conn_scores)
        conn_embedding = torch.matmul(self.conn_onehot_in_vocab,self.conn_roberta.roberta.embeddings.word_embeddings.weight)  # [conn_num, D]
        conn_embedding = conn_embedding / self.conn_length_in_vocab.unsqueeze(1)
        predict_embeds = torch.matmul(conn_scores, conn_embedding)  # [N, D], a soft connective embedding
        predict_embeds = predict_embeds.unsqueeze(1)  # [N, 1, D]

        ## 2 for relation classifiction
        # 2.1 prepare embeddings
        input_word_embeds = self.conn_roberta.roberta.embeddings.word_embeddings(input_ids)
        # input_word_embeds = torch.scatter(input_word_embeds, dim=1, index=mask_position_index, src=predict_embeds)

        # 2.2 roberta
        rel_outputs = self.conn_roberta.roberta(
            inputs_embeds=input_word_embeds,
            attention_mask=attention_mask
        )
        pooling_output = rel_outputs.last_hidden_state[:, 0, :]
        pooling_output = self.dropout(pooling_output)
        rel_logits = self.classifier(pooling_output)

        conn_preds = torch.argmax(conn_logits, dim=1)
        rel_preds = torch.argmax(rel_logits, dim=1)
        outputs = (conn_preds, rel_preds,)

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            conn_loss = loss_fct(conn_logits.view(-1, self.num_connectives), conn_ids.view(-1))
            rel_loss = loss_fct(rel_logits.view(-1, self.num_labels), labels.view(-1))
            loss = conn_loss + rel_loss
            outputs = (loss, conn_loss, rel_loss,) + outputs

        return outputs
