# -*- coding: utf-8 -*-
# VDocRetriever 官方源码中文注释版
# 来源：https://github.com/nttmdlab-nlp/VDocRAG/blob/main/src/vdocrag/vdocretriever/modeling/vdocretriever.py
# 本文件仅增加注释；使用时仍需放在官方项目及其依赖环境中。
#
# 阅读顺序：forward -> encode_query/encode_document -> _pooling -> compute_similarity/compute_loss。
# build/load 负责模型装配，_dist_gather_tensor 负责分布式收集。
# 与上一步衔接：TrainDataset 取样 -> TrainCollator 组批 -> Trainer 调用本模型。
# 模型负责计算向量、得分和 loss，反向传播与优化器更新由训练流程执行。
#
# dataclass 用于自动生成输出容器的初始化方法。
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

# AutoModelForCausalLM 按配置加载带语言建模头的底座；实际模型由模型路径和加载参数决定。
# 官方驱动传入的视觉语言模型可以同时处理文本输入与带像素张量的图像输入。
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq
# PEFT 工具用于配置、创建、恢复和合并 LoRA 适配器。
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from vdocrag.vdocretriever.arguments import ModelArguments, VDocRetrieverTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
# 统一的返回对象，可通过 outputs.q_reps、outputs.loss 等属性读取结果。
# 这里的 p 表示 passage/document，即页面文档；不要与 forward 的 pair 输入混淆。
class EncoderOutput(ModelOutput):
    # 文本检索向量，常见形状为 [文本条数, 向量维度]。
    q_reps: Optional[Tensor] = None
    # 页面检索向量，常见形状为 [页面条数, 向量维度]。
    p_reps: Optional[Tensor] = None
    # 训练损失；单独编码或评估分支中可以为 None。
    loss: Optional[Tensor] = None
    # 文本与页面两两之间的相似度矩阵，形状为 [文本条数, 页面条数]。
    scores: Optional[Tensor] = None

# 检索器主体，继承 PyTorch 的 nn.Module。
# 主要工作：编码文本与页面、池化成单向量、计算相似度、计算训练损失。
class VDocRetriever(nn.Module):
    # 保存底座模型的加载类，供 build() 和 load() 调用。
    TRANSFORMER_CLS = AutoModelForCausalLM 

    # 初始化检索器包装层。encoder 是已经创建的底座模型，并非模型名称字符串。
    # pooling 决定如何把 token 隐藏状态序列压缩成一个检索向量。
    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 ):
        # 初始化 nn.Module，使子模块、参数和训练/评估状态能被 PyTorch 管理。
        super().__init__()
        # 保存底座配置；encoder 本身作为子模块注册到当前模型。
        self.config = encoder.config
        self.encoder = encoder
        # 类中的默认值是 cls；论文使用的运行脚本可以通过参数选择 eos。
        self.pooling = pooling
        # 是否将最终检索向量做 L2 归一化。
        self.normalize = normalize
        # 计算检索交叉熵前，用温度参数缩放相似度分数。
        self.temperature = temperature
        # 输入是每个文本对所有候选页面的分数，以及正确页面所在的列下标。
        # CrossEntropyLoss 内部会处理 log-softmax，不需要在外面先调用 softmax。
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        # 检查分布式进程组是否已经初始化；后续据此决定是否跨进程收集向量。
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            # 当前进程编号，常用于区分不同设备上的数据。
            self.process_rank = dist.get_rank()
            # 参加分布式计算的进程总数。
            self.world_size = dist.get_world_size()

    # 模型调用 model(...) 时，由 nn.Module 调用此 forward()。
    # 与 TrainCollator 的对应关系：
    # query <- q_collated：文本 token ID 与普通注意力掩码。
    # document <- d_collated：图像相关 token、pixel_values、image_sizes 等。
    # pair <- p_collated：预训练时的图像提示+OCR 输入、labels 和可选定制掩码。
    # 这里的输入已经由 Collator 转成张量，forward 不再读取图片文件。
    def forward(self, query: Dict[str, Tensor] = None, 
                document: Dict[str, Tensor] = None, 
                pair: Dict[str, Tensor] = None, 
                use_cache: bool = True
        ):
        # 如果提供文本输入，计算文本检索向量；否则保留 None。
        q_reps = self.encode_query(query, use_cache=use_cache) if query else None
        # 如果提供页面输入，计算页面检索向量。
        p_reps = self.encode_document(document, use_cache=use_cache) if document else None
        # 如果提供预训练 pair，额外做一次带 labels 的语言建模前向计算。
        # outputs.loss 是 OCR 生成监督的损失；此处没有调用 generate() 自由生成答案。
        outputs = self.generate_output(pair, use_cache=use_cache) if pair else None # pre-training

        # for inference
        # 只有文本或只有页面时，直接返回相应向量；编码脚本就使用这条分支。
        # 这个分支根据输入是否齐全判断，不能简单理解为只在 model.eval() 时进入。
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        # model.train() 时进入训练分支；model.eval() 时 self.training 为 False。
        # 这段属性判断不会自行调用 optimizer，也不会自行更新参数。
        if self.training:
            if self.is_ddp:
                # 收集各进程的文本与页面向量，扩大本次损失计算使用的候选集合。
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            # 计算所有文本与所有页面的点积；每行对应一个文本，每列对应一个页面。
            scores = self.compute_similarity(q_reps, p_reps)
            # 将得分整理为 [文本数量, 候选页面数量] 的二维矩阵。
            scores = scores.view(q_reps.size(0), -1)

            # 先生成 [0, 1, 2, ...]；target 表示每个文本的正确页面位于哪一列。
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            # 按页面分组数量调整正确列下标，要求每组的第一个页面为正例。
            # 前面的 TrainDataset 一问配一图时，两侧条数相同，target 仍为 [0, 1, 2, ...]。
            # 因此在这类批次中，相似度矩阵的对角线是已配对正例，其余列作为批内负例使用。
            target = target * (p_reps.size(0) // q_reps.size(0))

            # 缩放相似度并计算检索交叉熵，促使正确页面在对应行中的分数更高。
            # 预训练时文本是 OCR，对应检索压缩监督；微调时文本通常是问题。
            loss = self.compute_loss(scores / self.temperature, target)

            # 存在预训练生成输出时，把额外的 OCR 生成损失加到检索损失上。
            if outputs:
                # 原实现直接相加，两个损失在这里没有额外可调权重。
                loss = loss + outputs.loss

            if self.is_ddp:
                # 保留官方实现的分布式损失缩放，具体训练步还要结合 trainer.py 阅读。
                loss = loss * self.world_size  # counter average weight reduction

        # for eval
        else:
            # 计算所有文本与所有页面的点积；每行对应一个文本，每列对应一个页面。
            scores = self.compute_similarity(q_reps, p_reps)
            # 评估分支仅返回相似度与向量，不在这里计算训练损失。
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    # 文本向量为 [Nq, H]，页面向量为 [Nd, H]，返回 [Nq, Nd]。
    def compute_similarity(self, q_reps, p_reps):
        # 将页面矩阵转置成 [H, Nd] 后做矩阵乘法，每个元素是一对向量的点积。
        # 若两侧向量都已经 L2 归一化，这个点积就等于余弦相似度。
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    # 计算单向的文本到页面检索交叉熵。
    # 这里没有另外计算页面到文本方向的交叉熵。
    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    # 开启底座的梯度检查点，以更多反向重算换取更低的激活存储开销。
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    # 辅助函数：把各进程中的同形状向量张量收集并拼接。
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        # 整理为连续内存布局，便于分布式通信。
        t = t.contiguous()

        # 为每个进程的结果分配一个相同形状的接收缓冲区。
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # 将每个进程的局部张量收集到 all_tensors 中。
        dist.all_gather(all_tensors, t)

        # 用本进程原始张量替换对应缓冲区，以保留本地向量的梯度连接。
        all_tensors[self.process_rank] = t
        # 沿样本维拼接，得到所有进程的向量集合。
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    # 训练阶段创建模型：加载底座，然后按配置新增或恢复可训练的 LoRA。
    # @classmethod 表示使用类本身 cls 调用，不需要先创建模型实例。
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        # 按模型名称/路径加载预训练底座；hf_kwargs 接收 dtype、缓存路径等额外参数。
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        # 底座没有设置 PAD ID 时，原代码把配置中的 PAD ID 设为 0。
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        # 如果开启 LoRA，或给出已有 LoRA 路径，就进入适配器分支。
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                # 为梯度检查点与适配器训练准备输入梯度；不等于解冻底座的全部参数。
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                # 恢复已有 LoRA，并设置适配器参数可继续训练。
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                # 创建新的 LoRA 配置，决定低秩大小、缩放、丢弃率和注入模块。
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    # 把 PEFT 任务类型配置为特征提取。
                    task_type=TaskType.FEATURE_EXTRACTION,
                    # LoRA 的低秩维度 r。
                    r=model_args.lora_r,
                    # LoRA 更新的缩放参数 alpha。
                    lora_alpha=model_args.lora_alpha,
                    # LoRA 分支使用的 dropout。
                    lora_dropout=model_args.lora_dropout,
                    # 把逗号分隔的模块名字符串拆成列表，指定要注入 LoRA 的线性层。
                    target_modules=model_args.lora_target_modules.split(','),
                    # 新建的适配器处于训练用途配置。
                    inference_mode=False
                )
                # 将新适配器安装到指定底座模块上。
                lora_model = get_peft_model(base_model, lora_config)
            # 调用本类构造函数，把准备好的底座包装为 VDocRetriever。
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
            )
        else:
            # 调用本类构造函数，把准备好的底座包装为 VDocRetriever。
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    # 编码/检索阶段加载模型，可选择加载已有 LoRA 并合并进底座。
    # 与 build() 的差别是这里会合并适配器；该方法本身不调用 model.eval()。
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        # 底座没有设置 PAD ID 时，原代码把配置中的 PAD ID 设为 0。
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            # 加载已经训练好的适配器。
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            # 将 LoRA 更新合并到基础权重，并移除适配器包装。
            lora_model = lora_model.merge_and_unload()
            # 调用本类构造函数，把准备好的底座包装为 VDocRetriever。
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            # 调用本类构造函数，把准备好的底座包装为 VDocRetriever。
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    # 将底层 encoder 按其 save_pretrained 接口保存到目录。
    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    # 共享的编码流程：底座前向计算 -> 取最后一层隐藏状态 -> 池化。
    # qry 是张量字典；**qry 会把它展开为 input_ids=...、attention_mask=... 等参数。
    def encode_query(self, qry, use_cache=True):
        # 请求模型返回隐藏状态；这里得到的是模型输出对象，还不是检索向量。
        # 原代码还请求注意力输出；后面真正用于检索的是 hidden_states。
        query_hidden_states = self.encoder(**qry, return_dict=True, output_hidden_states=True, output_attentions=True, use_cache=use_cache)
        # 取最后一层 token 隐藏状态，常见形状为 [B, L, H]。
        # B 是批次大小，L 是序列长度，H 是隐藏维度。
        query_hidden_states = query_hidden_states.hidden_states[-1]
        # 使用有效 token 掩码池化，将 [B, L, H] 压缩为 [B, H]。
        return self._pooling(query_hidden_states, qry['attention_mask'])
    
    # 页面和文本复用同一个 encoder 和同一套池化逻辑。
    # 区别在输入字典：页面输入额外带 pixel_values、image_sizes 等视觉数据。
    def encode_document(self, doc, use_cache=True):
        return self.encode_query(doc, use_cache=use_cache)

    # 预训练生成分支：pair 已由 Collator 准备好图像提示、OCR token 和 labels。
    # 直接调用底座前向计算，由其语言建模头计算 next-token 损失。
    # 这里生成的是训练输出对象，没有执行逐 token 自由解码。
    def generate_output(self, pair, use_cache=True):
        return self.encoder(**pair, use_cache=use_cache)

    # 把 token 序列转换成每个样本的单个检索向量。
    # 这里接收普通的 [B, L] 有效 token 掩码；预训练生成分支的四维掩码不走此函数。
    def _pooling(self, last_hidden_state, attention_mask):
        # 取第一个 token 的隐藏状态；名称 cls 不代表所有底座都含有专门的 CLS token。
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        # 对有效 token 的隐藏状态做平均，排除补齐位置。
        elif self.pooling in ['mean', 'avg', 'average']:
            # 掩码从 [B, L] 扩为 [B, L, 1]，按隐藏维广播，将 PAD 位置向量置零。
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            # 沿 token 维求和，再除以每条样本的有效 token 数，得到 [B, H]。
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        # 取最后一个有效 token。若输入已在末尾追加 EOS，就得到 EOS 的隐藏状态。
        # 这里根据长度定位，要求有效 token 位于左侧、PAD 补在右侧；原编码驱动采用右侧补齐。
        elif self.pooling in ['last', 'eos']:
            # 有效 token 数减 1 得到最后一个有效 token 的零起始下标。
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            # 为每条样本分别取其最后有效位置的 H 维向量。
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        # 若开启归一化，将每个检索向量缩放到单位 L2 长度。
        if self.normalize:
            # 只沿最后的向量维归一化，保持输出形状为 [B, H]。
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps
