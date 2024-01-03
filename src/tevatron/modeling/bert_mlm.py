from transformers import BertForMaskedLM
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
import json
import os
import copy
from sklearn.metrics import accuracy_score
from .madr import MADRPooler
from .mtbert import DensePooler
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertModel, BertLayer
import numpy as np
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from tevatron.loss import compute_aspect_loss


def load_dict(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_dict(dict_list, num=4):
    # 使用4级意图
    dict_list = dict_list[:num]
    all_dict = {}
    for num in range(len(dict_list)):
        cate_dict = dict_list[num]
        sorted(cate_dict.items(), key=lambda x: x[1])  # 根据id排序，固定遍历次序
        dict_len = len(cate_dict)
        idx = 0
        for cate, cate_id in cate_dict.items():
            assert idx == cate_id
            idx += 1
            if cate not in all_dict:
                all_dict[cate] = len(all_dict)
        assert dict_len == idx
    return all_dict


def cal_acc_for_mul(true_label, pred_logits, top_k=3):
    """
    每个样本topk的正确预测true_label的数量/true_label的总数量
    true_label: [bs, class_num] 可能有多个1
    pred_label: [bs, class_num]  每个类别的数目
    """
    topk_res = torch.topk(pred_logits, k=top_k, dim=1)  # topk的结果
    topk_class = topk_res.indices  # topk的位置 [bs, top_k]
    # 全0，根据topk构造和true_label对应的pred_label
    pred_label = np.zeros(pred_logits.shape, dtype=np.int64)

    index = np.arange(pred_logits.shape[0]).repeat(
        top_k)  # [0,0,0,1,1,1,..,bs-1,bs-1,bs-1]
    topk_class = topk_class.reshape(-1).cpu().numpy()  # [bs*top_k]
    pred_label[index, topk_class] = 1  # [bs, class_num] 索引，将topk设1
    same_num = np.sum((true_label.numpy() == pred_label).astype(
        int) * true_label.numpy())
    all_num = np.sum((true_label.numpy() == 1).astype(int))
    topk_acc = same_num / all_num
    return topk_acc


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    doc_loss_dict: Optional[dict] = None


class BertForMaskedLM_Same(BertForMaskedLM):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        # 3mlm test
        # masked_lm_loss = masked_lm_loss * 3
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MadrMaskLM(BertForMaskedLM):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        """
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        """
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.training_args = kwargs.pop('training_args')
        self.pretrain_aspect_alpha = self.training_args.pretrain_aspect_alpha
        model_args = kwargs.pop('model_args')
        self.aspect_loss_type = model_args.aspect_loss_type
        self.aspect_num = model_args.aspect_num

        self.tf_writer = kwargs.pop('tf_writer')
        data_args = kwargs.pop('data_args')
        aspect_id_dict = {
            'p_brand': load_dict(data_args.brand2id_path),
            'p_color': load_dict(data_args.color2id_path),
            'p_cate1': load_dict(data_args.cate1_2id_path),
            'p_cate2': load_dict(data_args.cate2_2id_path),
            'p_cate3': load_dict(data_args.cate3_2id_path),
            'p_cate4': load_dict(data_args.cate4_2id_path),
            'p_cate5': load_dict(data_args.cate5_2id_path),
        }
        all_cate_dict = [
            aspect_id_dict['p_cate1'],
            aspect_id_dict['p_cate2'],
            aspect_id_dict['p_cate3'],
            aspect_id_dict['p_cate4'],
            aspect_id_dict['p_cate5'],
        ]
        aspect_id_dict['p_cate_mul'] = merge_dict(all_cate_dict)

        aspect_lens_dict = {}
        for k in aspect_id_dict.keys():
            aspect_lens_dict[k] = len(aspect_id_dict[k])
        aspect_zeroid = {}
        for k in aspect_id_dict.keys():
            aspect_zeroid[k] = aspect_id_dict[k]['']
        self.aspect_zeroid = aspect_zeroid
        self.p_brand = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_brand'])
        self.p_color = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_color'])
        self.p_cate1 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate1'])
        self.p_cate2 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate2'])
        self.p_cate3 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate3'])
        self.p_cate4 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate4'])
        self.p_cate5 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate5'])
        self.p_cate_mul = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate_mul'])

        # alpha 为0的不计算loss
        self.doc_need_aspect_alpha = {
            'p_brand': 1,
            'p_color': 2,
            'p_cate1': 0,
            'p_cate2': 0,
            'p_cate3': 0,
            'p_cate4': 0,
            'p_cate5': 0,
            'p_cate_mul': 3,
        }
        print('======', self.doc_need_aspect_alpha)

        self.pooler = self.build_pooler(model_args)

        self.gating_type = model_args.gating_type
        if 'app' in self.gating_type:
            self.pooler_type = 'app'
        else:
            self.pooler_type = 'cls'
        print('=========pooler_type:', self.pooler_type)
        self.bce = nn.BCEWithLogitsLoss()

        # Initialize weights and apply final processing
        self.post_init()

        self.test = False
        if self.test:
            self.thred = model_args.thred  # 0 1 2,
            # <=0 取1/3;
            #  <=1 取 2/3;
            # <= 2 取全部
            print('================== self.thred: ', self.thred)

    def build_pooler(self, model_args):

        pooler = MADRPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            q_pool_type=model_args.q_pool_type,
            doc_pool_type=model_args.doc_pool_type,
            tied=not model_args.untie_encoder,
            aspect_num=model_args.aspect_num,
        )  # 如果前面共享的encoder，那么投影层就不共享；如果是两个encoder，就需要相同的投影层来到一个空间
        pooler.load(model_args.model_name_or_path)
        return pooler

    def sum_loss_dict(self, loss_dict):
        if loss_dict == {}:
            return None
        return sum(loss_dict.values())

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            aspect_dict=None,
            step=None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        for aspect_name, aspect_id_list in aspect_dict.items():
            aspect_dict[aspect_name] = aspect_id_list.to(input_ids.device)

        aspect_fc_dict = {
            'p_brand': self.p_brand,
            'p_color': self.p_color,
            'p_cate1': self.p_cate1,
            'p_cate2': self.p_cate2,
            'p_cate3': self.p_cate3,
            'p_cate4': self.p_cate4,
            'p_cate5': self.p_cate5,
            'p_cate_mul': self.p_cate_mul
        }
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [3*bs, seq_len, 768]
        cls_output = sequence_output[:, 0]  # [bs, dim]
        # [bs, seq_len, vocab_size]
        prediction_scores = self.cls(sequence_output)

        p_presence_for_aspect = {}
        if self.pooler_type == 'app':
            p_reps, doc_aspect_emb, doc_aspect_weight = self.pooler(
                doc=sequence_output, doc_attention_mask=attention_mask,
                need_aspect_alpha=self.doc_need_aspect_alpha, gating_type=self.gating_type)  # D * d
            doc_aspect_weight = doc_aspect_weight.to(self.device)
            for aspect_name, alpha in self.doc_need_aspect_alpha.items():
                if alpha != 0:
                    p_presence_for_aspect[aspect_name] = doc_aspect_weight[:, 0, alpha - 1]  # [160]
        else:
            p_reps, doc_aspect_emb, doc_aspect_weight = self.pooler(
                doc=sequence_output, doc_attention_mask=attention_mask, gating_type=self.gating_type)

        if self.test:
            mask = torch.randint(3, (p_reps.shape[0], 1))
            thred = self.thred  # 0 1 2, <=0 取1/3; <=1 取 2/3; <= 2 取全部
            need_mask_bool = (mask <= thred).squeeze()

            if self.thred >= 2:
                assert need_mask_bool.int().sum() == p_reps.shape[0]

        doc_app_loss_dict = {}
        doc_loss_dict = {}
        doc_acc_dict = {}
        doc_not_empty_aspect = 0  # 使用的aspect数量
        for doc_aspect_name, alpha in self.doc_need_aspect_alpha.items():
            if alpha != 0:
                doc_not_empty_aspect += 1
                aspect_rep = aspect_fc_dict[doc_aspect_name](
                    doc_aspect_emb[:, alpha - 1]).view(sequence_output.size(0), -1)
                aspect_label = aspect_dict[doc_aspect_name]
                aspect_loss_type = self.aspect_loss_type

                if self.pooler_type == 'app':
                    app_weight = p_presence_for_aspect[doc_aspect_name]  # [bs]

                if self.test:
                    # 只取少部分进行计算
                    aspect_rep = aspect_rep[need_mask_bool]
                    aspect_label = aspect_label[need_mask_bool]

                    if self.pooler_type == 'app':
                        app_weight = app_weight[need_mask_bool]

                if doc_aspect_name == 'p_cate_mul':
                    aspect_loss_type = 'multi-softmax'
                doc_loss_dict[doc_aspect_name] = compute_aspect_loss(
                    aspect_rep, aspect_label, aspect_loss_type,
                    zero_id=self.aspect_zeroid[doc_aspect_name]
                ).view(1)
                if doc_aspect_name == 'p_cate_mul':
                    doc_acc_dict[doc_aspect_name] = cal_acc_for_mul(
                        aspect_label.cpu(), aspect_rep.cpu().float())
                else:
                    doc_acc_dict[doc_aspect_name] = accuracy_score(
                        aspect_label.cpu(), aspect_rep.argmax(1).cpu().tolist())

                if self.pooler_type == 'app':
                    if doc_aspect_name == 'p_cate_mul':
                        app_label = (aspect_label.sum(1) != 0).int().float()
                    else:
                        app_label = (aspect_label !=
                                     self.aspect_zeroid[doc_aspect_name]).float()

                    doc_app_loss_dict[doc_aspect_name] = self.bce(
                        app_weight, app_label).view(1)

        # ==== 特殊注释
        if self.aspect_num != doc_not_empty_aspect + 1:
            raise ValueError('wrong aspect_num!')
        if doc_not_empty_aspect == 0:  # 针对aspect 数量取平均，防止被aspect 数量干扰alpha
            doc_not_empty_aspect = 1

        masked_lm_loss = None
        new_loss = masked_lm_loss

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.contiguous(
            ).view(-1, self.config.vocab_size), labels.contiguous().view(-1)).view(1)
            new_loss = masked_lm_loss.clone()

            aspect_predict_loss = self.sum_loss_dict(doc_loss_dict)
            doc_app_loss = self.sum_loss_dict(doc_app_loss_dict)

            if aspect_predict_loss is not None and self.pretrain_aspect_alpha != 0.0:
                aspect_predict_loss = aspect_predict_loss / doc_not_empty_aspect
                doc_loss_dict['avg_doc_loss'] = aspect_predict_loss

                new_loss += self.pretrain_aspect_alpha * aspect_predict_loss

            if doc_app_loss is not None and self.pretrain_aspect_alpha != 0.0:
                doc_app_loss = doc_app_loss / doc_not_empty_aspect
                new_loss += self.pretrain_aspect_alpha * doc_app_loss

                doc_loss_dict['app_loss'] = doc_app_loss

            # 最后加进去记录
            doc_loss_dict['mlm_loss'] = masked_lm_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if new_loss.device == torch.device('cuda:0') and self.tf_writer is not None:
            # 记录float类型指标，无法gather
            for k, v in doc_acc_dict.items():
                self.tf_writer.add_scalar(
                    'doc_acc_' + k, v, step)
        return MaskedLMOutput(
            loss=new_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            doc_loss_dict=doc_loss_dict,
        )

    def additional_save(self, output_dir: str):
        fc_dict = {
            'brand_fc': self.p_brand,
            'color_fc': self.p_color,
            'cate1_fc': self.p_cate1,
            'cate2_fc': self.p_cate2,
            'cate3_fc': self.p_cate3,
            'cate4_fc': self.p_cate4,
            'cate5_fc': self.p_cate5,
            'cate_mul_fc': self.p_cate_mul,
        }
        for fc_name, fc_layer in fc_dict.items():
            if fc_layer:
                torch.save(fc_layer.state_dict(),
                           os.path.join(output_dir, '{}.pt'.format(fc_name)))

        if self.pooler is not None:
            self.pooler.save_pooler(output_dir)


class MtbertMaskLM(BertForMaskedLM):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        """
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        """
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.training_args = kwargs.pop('training_args')
        self.pretrain_aspect_alpha = self.training_args.pretrain_aspect_alpha
        model_args = kwargs.pop('model_args')
        self.aspect_loss_type = model_args.aspect_loss_type

        self.tf_writer = kwargs.pop('tf_writer')
        data_args = kwargs.pop('data_args')
        aspect_id_dict = {
            'p_brand': load_dict(data_args.brand2id_path),
            'p_color': load_dict(data_args.color2id_path),
            'p_cate1': load_dict(data_args.cate1_2id_path),
            'p_cate2': load_dict(data_args.cate2_2id_path),
            'p_cate3': load_dict(data_args.cate3_2id_path),
            'p_cate4': load_dict(data_args.cate4_2id_path),
            'p_cate5': load_dict(data_args.cate5_2id_path),
            # 'p_cate_mul': load_dict(data_args.whole_cate_vocab_file),
        }
        all_cate_dict = [
            aspect_id_dict['p_cate1'],
            aspect_id_dict['p_cate2'],
            aspect_id_dict['p_cate3'],
            aspect_id_dict['p_cate4'],
            aspect_id_dict['p_cate5'],
        ]
        aspect_id_dict['p_cate_mul'] = merge_dict(all_cate_dict)

        aspect_lens_dict = {}
        for k in aspect_id_dict.keys():
            aspect_lens_dict[k] = len(aspect_id_dict[k])
        aspect_zeroid = {}
        for k in aspect_id_dict.keys():
            if '' in aspect_id_dict[k]:
                aspect_zeroid[k] = aspect_id_dict[k]['']
            else:
                aspect_zeroid[k] = None
        self.aspect_zeroid = aspect_zeroid
        self.p_brand = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_brand'])
        self.p_color = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_color'])
        self.p_cate1 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate1'])
        self.p_cate2 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate2'])
        self.p_cate3 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate3'])
        self.p_cate4 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate4'])
        self.p_cate5 = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate5'])
        self.p_cate_mul = nn.Linear(
            config.hidden_size, aspect_lens_dict['p_cate_mul'])

        # alpha 为0的不计算loss
        self.doc_need_aspect_alpha = {
            'p_brand': 1,
            'p_color': 1,
            'p_cate1': 0,
            'p_cate2': 0,
            'p_cate3': 0,
            'p_cate4': 0,
            'p_cate5': 0,
            'p_cate_mul': 1,
        }
        print('========== aspect num:', self.doc_need_aspect_alpha)

        # self.pooler = None
        self.pooler = self.build_pooler(model_args)
        print('======mtbert, have pooler')

        # Initialize weights and apply final processing
        self.post_init()
    
    def build_pooler(self, model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    def sum_loss_dict(self, loss_dict):
        all_loss = None
        for k, v in loss_dict.items():
            if all_loss is not None:
                all_loss += v
            else:
                all_loss = v.clone()
        return all_loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        aspect_dict=None,
        step=None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        for aspect_name, aspect_id_list in aspect_dict.items():
            aspect_dict[aspect_name] = aspect_id_list.to(input_ids.device)

        aspect_fc_dict = {
            'p_brand': self.p_brand,
            'p_color': self.p_color,
            'p_cate1': self.p_cate1,
            'p_cate2': self.p_cate2,
            'p_cate3': self.p_cate3,
            'p_cate4': self.p_cate4,
            'p_cate5': self.p_cate5,
            'p_cate_mul': self.p_cate_mul,
        }

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [3*bs, seq_len, 768]
        cls_output = sequence_output[:, 0]  # [bs, dim]

        if self.pooler is not None:
            cls_output = self.pooler(q=sequence_output)  # sequence_output[:,0] == cls_output
        prediction_scores = self.cls(sequence_output)

        doc_loss_dict = {}
        doc_acc_dict = {}
        doc_not_empty_aspect = 0  # 使用的aspect数量

        for doc_aspect_name, alpha in self.doc_need_aspect_alpha.items():
            if alpha != 0:
                doc_not_empty_aspect += 1
                aspect_rep = aspect_fc_dict[doc_aspect_name](
                    cls_output).view(cls_output.size(0), -1)
                aspect_label = aspect_dict[doc_aspect_name]
                aspect_loss_type = self.aspect_loss_type
                if doc_aspect_name == 'p_cate_mul':
                    aspect_loss_type = 'multi-softmax'
                    # doc_loss_dict[doc_aspect_name] = multi_label_loss_fct(
                        # aspect_rep, aspect_label)
                doc_loss_dict[doc_aspect_name] = compute_aspect_loss(
                    aspect_rep, aspect_label, aspect_loss_type,
                    zero_id=self.aspect_zeroid[doc_aspect_name])
                if doc_aspect_name == 'p_cate_mul':
                    doc_acc_dict[doc_aspect_name] = cal_acc_for_mul(
                        aspect_label.cpu(), aspect_rep.cpu().float())
                else:
                    doc_acc_dict[doc_aspect_name] = accuracy_score(
                        aspect_label.cpu(), aspect_rep.argmax(1).cpu().tolist())
        if doc_not_empty_aspect == 0:  # 针对aspect 数量取平均，防止被aspect 数量干扰alpha
            doc_not_empty_aspect = 1

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.contiguous(
            ).view(-1, self.config.vocab_size), labels.contiguous().view(-1))

            new_loss = masked_lm_loss.clone()

            aspect_predict_loss = self.sum_loss_dict(doc_loss_dict)
            if aspect_predict_loss is not None:
                aspect_predict_loss = aspect_predict_loss / doc_not_empty_aspect
                new_loss += self.pretrain_aspect_alpha * aspect_predict_loss
                doc_loss_dict['avg_doc_loss'] = aspect_predict_loss

            # 最后加进去记录
            doc_loss_dict['mlm_loss'] = masked_lm_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if new_loss.device == torch.device('cuda:0') and self.tf_writer is not None:
            # 记录float类型指标，无法gather
            for k, v in doc_acc_dict.items():
                self.tf_writer.add_scalar(
                    'doc_acc_' + k, v, step)
        return MaskedLMOutput(
            loss=new_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            doc_loss_dict=doc_loss_dict,
        )

    def additional_save(self, output_dir: str):
        fc_dict = {
            'brand_fc': self.p_brand,
            'color_fc': self.p_color,
            'cate1_fc': self.p_cate1,
            'cate2_fc': self.p_cate2,
            'cate3_fc': self.p_cate3,
            'cate4_fc': self.p_cate4,
            'cate5_fc': self.p_cate5,
            'cate_mul_fc': self.p_cate_mul,
        }
        for fc_name, fc_layer in fc_dict.items():
            if fc_layer:
                torch.save(fc_layer.state_dict(),
                           os.path.join(output_dir, '{}.pt'.format(fc_name)))

        if self.pooler is not None:
            self.pooler.save_pooler(output_dir)


class CondenserForPretraining(BertForMaskedLM):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config, **kwargs):
        # https://github.com/luyug/Condenser/blob/main/modeling.py
        super().__init__(config)
        model_args = kwargs.pop('model_args')

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        n_head_layers = model_args.head_num  # 额外的head有几层
        self.skip_from = model_args.skip_from  # 从几层短连
        self.late_mlm = True

        self.c_head = nn.ModuleList(
            [BertLayer(self.bert.config) for _ in range(n_head_layers)]
        )
        self.c_head.apply(self.bert._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def new_post_init(self):
        self.c_head.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # outputs.hidden_states[-1]
        ori_prediction_scores = self.cls(sequence_output)
        ori_loss = self.mlm_loss(ori_prediction_scores, labels)

        # 新的
        cls_hiddens = outputs.hidden_states[-1][:, :1]
        skip_hiddens = outputs.hidden_states[self.skip_from]

        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            attention_mask.shape,
            attention_mask.device
        )  # [batch_size, 1, 1, seq_length]
        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                extended_attention_mask,
            )
            hiddens = layer_out[0]

        prediction_scores = self.cls(hiddens)

        loss = self.mlm_loss(prediction_scores, labels)
        if self.late_mlm:
            loss += ori_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=ori_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def mlm_loss(self, pred_scores, labels):
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss