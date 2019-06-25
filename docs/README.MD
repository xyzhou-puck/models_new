# Paddle官方模型库开发规范

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/PaddlePaddle/models)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

## 目录
- [**整体目标和原则**](#整体目标和原则)

- [**必选规范**](#必选规范)

  - [**目录结构**](#目录结构)
  
  - [**功能实现**](#功能实现)
  
  - [**书写示例**](#书写示例)
  
  - [**命名规范**](#命名规范)
  
  - [**README**](#README)
  
  - [**多环境支持**](#多环境支持)
  
  - [**注释和Licenses**](#注释和Licenses)

- [**建议规范**](#建议规范)

  - [**工具包PALM**](#工具包PALM)
  
  - [**显存优化**](#显存优化)
  
- [**合入规则**](#合入规则)
  
  - [**代码提交**](#代码提交)
  
  - [**code_review**](#code_review)

- [**升级维护**](#升级维护)

## 整体目标和原则

### 整体目标
为PaddlePaddle/models用户，提供功能一致，风格统一，可插拔的NLP，CV，Speech等领域的重要模型。
提高PaddlePaddle用户的开发效率。
加速部分老旧有bug的API离场，并推进新API的快速落地。

### 原则
1. 不同领域在逻辑上统一功能，在具体落实上（如configure的使用方式），尊重并拥抱各个领域的主流做法。但在具体大方向上，不可进一步拆分，需形成统一。
2. 区分规范为必选和可选，原则上要求所有模型遵守必选规范，建议遵守可选规范。
3. 原则上要求模型可以在**GPU单卡多卡**，**CPU多核**上执行**训练**，**预测**，**评估**和**准备部署环境**，并可以支持轻松**使用自定义数据**。
4. 如果有特殊情况，需要在readme中的显眼位置做出说明，避免用户踩坑。

## 必选规范

### 目录结构

遵照1.5 Paddle-models结构，将整体模型库拆分为3级目录结构，如下图所示：

#### 一级目录，按照大方向进行细分，如PaddleNLP，PaddleCV等：

![avatar](./appendix/dir-1.png)

#### 二级目录，按照具体场景进行拆分，如在PaddleNLP下，有neural_machine_translation，reading_comprehension等：

![avatar](./appendix/dir-2.png)

#### 三级目录，按照具体任务进行拆分，如在reading_comprehension下，有squad任务，mrqa任务等：

![avatar](./appendix/dir-3.png)

#### 叶子目录，具体到某个具体的任务，如SQuAD，则建议包涵以下内容：

![avatar](./appendix/dir-4.png)

- **data**：存储所有跟数据相关的内容，如configure，dict，input，output，saved_models，inferenece_models等。
- **squad**：该文件夹命名可以根据具体任务进行命名，保存该任务下一些特定的代码，如reader.py，batching.py，tokenizer.py等。
- **README.MD**: 当前任务库的使用手册，规范参见[**使用手册规范**](#README)
- **XXX_net.py**: 保存网络定义的脚本，可以有多个XXX_net.py，不同net之间在网络声明阶段需要统一output，也需要尽量统一input。
- **eval.py**：执行评估的脚本。
- **inference_model.py**：执行保存inference_model的脚本，用于准备上线部署环境。
- **main.py**：主程序入口，通过命令行或者configure文件，执行不同逻辑。
- **predict.py**：执行预测功能的脚本，需要可以单独运行。
- **train.py**：执行训练功能的脚本，需要可以单独运行。
- **run.sh**：示例脚本，需要给出不同环境下（GPU单卡多卡，CPU多核），如何进行训练，预测，评估，保存部署环境等功能。

### 功能实现

原则上需要官方提供的模型需要提供：
1. **训练**：可以在GPU单卡/多卡，CPU多核的环境下执行训练，至少有一个环境可以打平竞品效果或者复现原有paper效果（如果存在竞品），官方可以给出一种推荐训练的方式。
2. **预测**：可以在GPU单卡和CPU单核下执行预测。
3. **评估**：要求同预测。
4. **保存部署环境**：可以保存可以部署环境，支持模型部署。
5. **使用自定义数据**：要求官方模型可以灵活支持/适配用户自定义数据，可以通过在readme中加入数据格式描部分和**如何使用自定义数据**章节解决。

### 书写示例

下面以SQUAD下面的ERNIE-based MRC模型具体说明一些代码书写方面的问题。

#### net.py书写示例
```python

def create_net(
  is_training, 
  model_input, 
  args):
  
    """
    create the network of BERT-based Machine Reading Comprehension Network.
    is_training: a boolean value, indicating if the model to be created is for training or predicting.
    model_input: stores the input of the model
      for simple task like mnist: model_input can be either a single/tuple/list of fluid.layers.data.
      for complicated task like ERNIE-based machine reading comprehension, the model_input is an object like:
      
      class model_input:
        def __inint__(self):
          self.image = fluid.layers.data(shape = [-1, 8, 8, 1], dtype = "int64", name = "image")
      
      you can define your own one in the code or you can use PALM for easy input data management.
      
    """

    # declare the model input here
    if is_training:
        src_ids = model_input.src_ids
        ...
        is_null_answer = model_input.is_null_answer
    else:
        src_ids = model_input.src_ids
        ...
        unique_id = model_input.unique_id

    # define the model bert first

    assert isinstance(args.bert_config_path, str)

    # define the model here
    bert_conf = JsonConfig(args.bert_config_path)
    base_model = BertModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=bert_conf)

    sequence_output = base_model.get_sequence_output()

    ...

    # if is_training, then return the loss, otherwise return the prediction

    if is_training:

        def compute_loss(logits, positions):
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions)
            loss = fluid.layers.mean(x=loss)
            return loss

        start_loss = compute_loss(start_logits, start_positions)
        end_logits = fluid.layers.reshape(
            x=end_logits,
            shape=[-1, args.start_top_k, list(end_logits.shape)[-1]])
        end_logits = fluid.layers.slice(
            end_logits, axes=[1], starts=[0], ends=[1])
        end_logits = fluid.layers.reshape(
            x=end_logits, shape=[-1, list(end_logits.shape)[-1]])
        end_loss = compute_loss(end_logits, end_positions)

        total_loss = (start_loss + end_loss) / 2.0

        if args.use_fp16 and args.loss_scaling > 1.0:
            total_loss = total_loss * args.loss_scaling

        return total_loss

    else:

        predict = [unique_id, top_k_start_log_probs, top_k_start_indexes, \
            top_k_end_log_probs, top_k_end_indexes]

        return predict

```

#### train.py书写示例

#### predict.py书写示例

#### eval.py书写示例

#### main.py书写示例

### 命名规范
1. 文件和文件夹命名

### README


### 多环境支持


### 注释和Licenses


## 建议规范


