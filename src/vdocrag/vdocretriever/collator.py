import torch
import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, ProcessorMixin
from vdocrag.vdocretriever.arguments import DataArguments
from collections import defaultdict

logger = logging.getLogger(__name__)


from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

from vdocrag.vdocretriever.arguments import DataArguments


@dataclass
class TrainCollator:
    # 数据处理配置，例如长度上限、预训练开关、EOS 开关。
    data_args: DataArguments
    # 处理文本分词和 padding。
    tokenizer: PreTrainedTokenizer
    # 处理页面图像和多模态提示词。
    processor: ProcessorMixin
    def build_image_attention_mask(self, seq_len, input_lengths):
        """构造预训练生成分支的注意力掩码。
        参数
            seq_len：批次内补齐后的完整序列长度。
            input_lengths：每条样本的页面提示词长度。
        返回
            [batch_size, seq_len, seq_len]。
            1 表示允许读取，0 表示遮挡。
        """
        image_attention_masks = []
        for input_len in input_lengths:
            # 建立下三角因果掩码。
            # 每个位置可以读取自己和前面的位置。
            image_attention_mask = torch.tril(
                torch.ones(seq_len, seq_len),
                diagonal=0,
            )
            # OCR 部分从 input_len 开始。
            # 遮挡它对页面提示词前 input_len-1 个位置的直接访问。
            # 保留页面提示词最后一个位置，官方配置下该位置为 EOS。
                  
        # 拼成批次掩码，得到 [B, L, L]。
        image_attention_masks = torch.cat(
            image_attention_masks,
            dim=0,
        )

        return image_attention_masks

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """把一批文本与页图整理成模型输入。

        实际每条样本为 (文本, 图像)。
        普通微调的文本是问题，预训练的文本是 OCR。

        返回
            q_collated：文本侧输入。
            d_collated：页图侧输入。
            p_collated：OCR 生成输入，普通微调时为 None。
        """

        # 取出这一批的文本和图像。
        all_queries = [f[0] for f in features]
        all_images = [f[-1] for f in features]

        # 文本侧先分词，稍后统一 padding。
        q_collated = self.tokenizer(
            all_queries,
            padding=False,
            truncation=True,

            # 准备追加 EOS 时，为它预留一个位置。
            max_length=(
                self.data_args.query_max_len - 1
                if self.data_args.append_eos_token
                else self.data_args.query_max_len
            ),

            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
            #return_attention_mask=Fals	此时先不返回注意力掩码
            #return_token_type_ids=False	不返回用于区分句子片段的编号
            #add_special_tokens=True	按 tokenizer 的配置加入特殊 token，具体加入哪些由 tokenizer 决定
        )

        # 每张页图分别与相同的提示词一起交给 processor。
        collated_list = [
            self.processor(
                "<|image_1|>\nWhat is shown in this image?",
                image,
                return_tensors="pt",
            )
            for image in all_images
        ]

        # 取各页的 token ID。
        # 单页处理结果带一个批量维度，[0] 将它去掉。
        d_collated = {}
        d_collated["input_ids"] = [
            d["input_ids"][0].tolist()
            for d in collated_list
        ]

        if self.data_args.append_eos_token:
            # 给文本侧和页图侧的序列末尾分别追加 EOS。
            q_collated["input_ids"] = [
                q + [self.tokenizer.eos_token_id]
                for q in q_collated["input_ids"]
            ]

            d_collated["input_ids"] = [
                d + [self.tokenizer.eos_token_id]
                for d in d_collated["input_ids"]
            ]

        if self.data_args.pretrain:
            # 预训练时，额外构造“页面提示词＋OCR 目标”。
            p_collated = {}
            all_input_ids = []
            all_label_ids = []
            input_lengths = []

            for i, ocr in enumerate(all_queries):
                # 页面提示词，形状为 [1, P]。
                prompt_input_ids = torch.tensor(
                    d_collated["input_ids"][i]
                ).unsqueeze(0)

                # OCR 文本与结束标记。
                answer = f"{ocr}<|end|>\n<|endoftext|>"

                # 目标 token，形状为 [1, A]。
                answer_input_ids = self.tokenizer(
                    answer,
                    add_special_tokens=False,
                    max_length=self.data_args.answer_max_len,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]

                # 完整输入为 页面提示词＋OCR 目标。
                # [1, P]＋[1, A] → [1, P+A]。
                input_ids = torch.cat(
                    [prompt_input_ids, answer_input_ids],
                    dim=1,
                )

                # 页面提示词的标签设为 -100，跳过这些位置的损失。
                # OCR 和结束标记使用真实 token ID 作为标签。
                labels = torch.cat(
                    [
                        torch.tensor(
                            [-100] * len(prompt_input_ids[0])
                        ).unsqueeze(0),
                        answer_input_ids,
                    ],
                    dim=1,
                )

                # [1, L] → [L, 1]，准备交给 pad_sequence。
                all_input_ids.append(
                    input_ids.squeeze(0).unsqueeze(1)
                )
                all_label_ids.append(
                    labels.squeeze(0).unsqueeze(1)
                )

                # 记录页面提示词长度，供注意力掩码使用。
                input_lengths.append(
                    len(prompt_input_ids[0])
                )

            # 将生成输入补到批次内最长长度。
            # [B, Lmax, 1] → [B, Lmax]。
            input_ids = torch._C._nn.pad_sequence(
                all_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ).squeeze(2)

            # 标签的填充位置也设为 -100。
            labels = torch._C._nn.pad_sequence(
                all_label_ids,
                batch_first=True,
                padding_value=-100,
            ).squeeze(2)

            p_collated["input_ids"] = input_ids
            p_collated["labels"] = labels

            if self.data_args.image_attention_mask:
                image_attention_mask = (
                    self.build_image_attention_mask(
                        input_ids.size()[1],
                        input_lengths,
                    )
                )

                # [B, L, L] → [B, 1, L, L]。
                # 中间维度可以广播到各个注意力头。
                p_collated["attention_mask"] = (
                    image_attention_mask.unsqueeze(1)
                )
        else:
            # 普通微调只有文本和页图两路输入。
            p_collated = None

        # 文本侧补齐长度，生成 [B, Lq] 的 token 和掩码。
        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # 页图提示词补齐长度，生成 [B, Ld] 的 token 和掩码。
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # 将单页图像张量堆叠成批次图像输入。
        d_collated["pixel_values"] = torch.stack(
            [d["pixel_values"][0] for d in collated_list],
            dim=0,
        )

        # 堆叠每张页图的尺寸信息。
        d_collated["image_sizes"] = torch.stack(
            [d["image_sizes"][0] for d in collated_list],
            dim=0,
        )

        if self.data_args.pretrain:
            # 生成分支复用相同的页图张量和尺寸信息。
            p_collated["pixel_values"] = d_collated["pixel_values"]
            p_collated["image_sizes"] = d_collated["image_sizes"]

        return q_collated, d_collated, p_collated
@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, str]]):
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        images = [x[-1] for x in features]

        if self.data_args.encode_is_query:
            collated = self.tokenizer(
                texts,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
        else:
            collated = {}
            collated_list = [self.processor("<|image_1|>\nWhat is shown in this image?", image, return_tensors="pt") for image in images]
            collated['input_ids'] = [d['input_ids'][0].tolist() for d in collated_list]

        if self.data_args.append_eos_token:
            collated['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated['input_ids']]
            
        collated = self.tokenizer.pad(
            collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        if not self.data_args.encode_is_query:
            collated['pixel_values'] = torch.stack([d['pixel_values'][0] for d in collated_list], dim=0)
            collated['image_sizes'] = torch.stack([d['image_sizes'][0] for d in collated_list], dim=0)
        
        return text_ids, collated
