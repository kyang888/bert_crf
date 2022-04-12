# -*- coding: utf-8 -*-
import os
import torch
from dataset.dataset import NERDataset
from model.optimization import AdamW, get_linear_schedule_with_warmup
from common_utils.utils import restore_from_checkpoint, save_pytorch_model, save_best_model
from common_utils.tokenization import FullTokenizer
import random
import logging
import time
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import json
from model.modeling_bert import BertForNER
from config.arguments import parser
from config.hyper_parameters import HyperParams
from metrics.ner_metrics import SeqEntityScore


def valid(hp, global_step, model, device, writer, file_data, batch_size=1, mode="dev"):
    metric = SeqEntityScore()
    dataset = NERDataset(file_data, hp)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size, drop_last=False,
                            collate_fn=dataset.collate_wrapper)
    logging.info(f"mode: {mode}, eval batch size: {batch_size}, datasize: {len(dataset)}, batch num: {len(dataloader)}")

    loss = 0.0

    ner2id = json.load(open(hp.ner_file, "r", encoding="utf-8"))
    id2ner = dict([(v,k) for k,v in ner2id.items()])
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):

            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            loss_batch, preds = model.predict(input_ids, labels, batch['seqs_length'], attention_mask=attention_mask)
            if loss_batch:
                loss += loss_batch.item()

            for idx in range(input_ids.shape[0]):
                seq_len = torch.sum(attention_mask[idx])
                pred = [id2ner[i.item()] for i in preds[idx][0:seq_len]]
                label = [id2ner[i.item()] for i in labels[idx][0:seq_len]]
                metric.update([label], [pred])

        loss /= (step + 1)
        index, _ = metric.result()
        logging.info(
            f"{mode} end, f1: {index['f1']:0.4f}, acc: {index['acc']:0.4f}, recall: {index['recall']:0.4f}, loss: {loss / (step + 1): 0.5f}")
        writer.add_scalar(f"{mode}/acc", index['acc'], global_step)
        writer.add_scalar(f"{mode}/recall", index['recall'], global_step)
        writer.add_scalar(f"{mode}/f1", index['f1'], global_step)
        writer.add_scalar(f"{mode}/loss", loss / (step + 1), global_step)
    model.train()
    return index['acc'], index['recall'], index['f1'], loss


def train(model, optimizer, scheduler, global_step, writer, device, hp, dataset_train, dataloader_train):
    loss_interval = 0.0
    while global_step < hp.num_train_step:
        logging.info(f"train data size: {len(dataset_train)}, dataloader size: {len(dataloader_train)}")
        start_time = time.time()
        model.train()
        for step, batch in enumerate(dataloader_train):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            loss_batch = model(input_ids,labels,batch['seqs_length'],attention_mask=attention_mask)

            # if sum(start_labels) > 0:

            if hp.n_gpu > 1:
                loss_batch = loss_batch.mean()

            if hp.gradient_accumulation_steps > 1:
                loss_batch = loss_batch / hp.gradient_accumulation_steps

            loss_batch.backward()
            loss_interval += loss_batch.item()

            if (step + 1) % hp.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                if global_step % hp.save_checkpoint_interval:
                    with torch.no_grad():
                        writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)

                # 每隔 step_logging.info_info 打印一次信息
                if global_step % hp.interval_print_info == 0:
                    loss_interval = loss_interval / hp.interval_print_info

                    speed_global_step = (time.time() - start_time) / hp.interval_print_info
                    lr = scheduler.get_lr()[0]
                    logging.info(
                        f"global_step: {global_step}, loss: {loss_interval:0.5f}, "
                        f"speed: {speed_global_step: 0.2f}s/step, lr: {lr :0.7f}")
                    writer.add_scalar("train/loss", loss_interval, global_step)
                    loss_interval, start_time = 0.0, time.time()


                if global_step % hp.save_checkpoint_interval == 0:
                    # 在dev上验证
                    acc_dev, recall_dev, f1_dev, loss_dev = valid(hp,global_step,model,device,writer,os.path.join(hp.dir_training_data,"dev.json"),
                                                                   hp.batch_size_dev,mode="dev")

                    index_dict = {"acc": acc_dev, "recall": recall_dev, "f1": f1_dev,
                                  "loss": loss_dev}
                    save_best_model(global_step, model, optimizer, scheduler, index_dict, "f1", hp)
                    save_pytorch_model(hp.dir_checkpoint, model, f"pytorch_model_{global_step}.pt",
                                       hp.max_num_checkpoint,
                                       "model")
                    save_pytorch_model(hp.dir_checkpoint, optimizer, f"pytorch_optimizer_{global_step}.pt",
                                       hp.max_num_checkpoint, "optimizer")
                    save_pytorch_model(hp.dir_checkpoint, scheduler, f"pytorch_scheduler_{global_step}.pt",
                                       hp.max_num_checkpoint, "scheduler")

                    # test上可能没有标签
                    # acc_test, loss_test = valid(global_step, model, device, writer,
                    #                            os.path.join(hp.dir_training_data, "test.csv"), hp.batch_size_dev,
                    #                            mode="test")
                    # writer.add_scalar("test/acc", acc_test, global_step)
                    # writer.add_scalar("test/loss", loss_test, global_step)
                if global_step >= hp.num_train_step:
                    logging.info(f"training end!!!")
                    break


def device_config(hp):
    if hp.local_rank == -1 or hp.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not hp.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(hp.local_rank)
        device = torch.device("cuda", hp.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(hp.local_rank != -1), hp.fp16))
    return device, n_gpu


def random_seed_config(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def main(hp):
    # 获取设备
    device, n_gpu = device_config(hp)

    hp.n_gpu = n_gpu
    # 配置种子
    random_seed_config(hp.seed, n_gpu)
    model = BertForNER(hp)
    model.to(device)

    # from model.test import BertQueryNerConfig, BertQueryNER
    # /root/autodl-tmp/ner/hagongda
    # bert_config = BertQueryNerConfig.from_pretrained("/root/autodl-tmp/ner/init",
    #                                                      hidden_dropout_prob=0.1,
    #                                                      attention_probs_dropout_prob=0.1,
    #                                                      mrc_dropout=0.2,
    #                                                      classifier_act_func = "gelu",
    #                                                      classifier_intermediate_hidden_size=768 * 2)

    # model = BertQueryNER.from_pretrained("/root/autodl-tmp/ner/init", config=bert_config)
    # model.to(device)

    if os.path.exists(hp.dir_init_checkpoint):
        logging.info(f"init from {hp.dir_init_checkpoint}")
        state_dict = torch.load(hp.dir_init_checkpoint, map_location="cpu")
        miss_expect = model.load_state_dict(state_dict, strict=False)
        print(miss_expect)

    global_step = 0
    writer = SummaryWriter(log_dir=str(hp.dir_summary))
    # 恢复模型参数
    if os.path.exists(os.path.join(hp.dir_checkpoint, "model")):
        global_step = restore_from_checkpoint(model, hp.dir_checkpoint, hp.map_location, strict=True, type="model")
        logging.info(f"in {hp.dir_checkpoint}, model init from {global_step}")
    else:
        global_step = 0
        save_pytorch_model(hp.dir_checkpoint, model, "pytorch_model_0.pt", max_save=hp.max_num_checkpoint,
                           type="model")



    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": hp.weigth_decay_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    file_train_data = os.path.join(hp.dir_training_data, "train.json")
    dataset_train = NERDataset(file_train_data, hp)
    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train),
                                  batch_size=hp.batch_size_train, drop_last=False,
                                  collate_fn=dataset_train.collate_wrapper)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),  # according to RoBERTa paper
                      lr=hp.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(hp.warmup_epoch * len(dataloader_train)), num_training_steps=hp.num_train_step
    )



    # 恢复优化器
    record_checkpoint_optimizer = os.path.join(hp.dir_checkpoint, "optimizer")
    if os.path.exists(record_checkpoint_optimizer):
        global_step = restore_from_checkpoint(optimizer, hp.dir_checkpoint, type="optimizer")
        logging.info(f"in {hp.dir_checkpoint}, optimizer init from {global_step}")
    # 恢复scheduler
    record_checkpoint_scheduler = os.path.join(hp.dir_checkpoint, "scheduler")
    if os.path.exists(record_checkpoint_scheduler):
        global_step = restore_from_checkpoint(scheduler, hp.dir_checkpoint, type="scheduler")
        logging.info(f"in {hp.dir_checkpoint}, scheduler init from {global_step}")
    model.train()
    optimizer.zero_grad()
    train(model, optimizer, scheduler, global_step, writer, device, hp, dataset_train, dataloader_train)


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    hp = HyperParams.init_from_parsed_args(parsed_args)
    os.makedirs(hp.dir_checkpoint, exist_ok=True)
    # 开源的checkpoint
    hp.do_random_next = True
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level="INFO",
                        filename=os.path.join(hp.dir_log, "log.txt"), filemode='a')
    json.dump(vars(hp), open(os.path.join(hp.dir_checkpoint, "model_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    logging.info(hp.to_json_string())

    main(hp)
