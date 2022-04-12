import logging
import os
import re
import torch
import json

def restore_from_checkpoint(model, dir_checkpoint, map_location="cpu", strict=True, type="model"):
    """Restore model or optimizer from checkpoint.
        Args:
            model: the model or optimizer object to restore
            dir_checkpoint: a directory containing checkpoints and recording file.
            map_location: a :classification:`torch.device` object or a string contraining a device tag, it
                          indicates the location where all tensors should be loaded.
            strict: whether or not the weight in model should strictly be restored from checkpoint.
            type: string, either 'model' or 'optimizer'

        Returns:
            global_step: the global step of restored checkpoint.
            """
    if type == "model":
        pattern = re.compile(r"pytorch_model_([0-9].*)\.pt")
    elif type == "optimizer":
        pattern = re.compile(r"pytorch_optimizer_([0-9].*)\.pt")
    elif type == "scheduler":
        pattern = re.compile(r"pytorch_scheduler_([0-9].*)\.pt")
    else:
        raise ValueError("Argument `type` must be either `model` or `optimizer`.")

    record_checkpoint_model = os.path.join(dir_checkpoint, type)
    with open(record_checkpoint_model, "r", encoding="utf-8") as r:
        file_pts = r.readlines()
    restore_pt = file_pts[-1].strip()
    match = re.search(pattern, restore_pt)
    global_step = int(match.groups()[0])

    if type == "model":
        state_dict = torch.load(restore_pt, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
    elif type == "optimizer":
        state_dict = torch.load(restore_pt)
        model.load_state_dict(state_dict)
    elif type == "scheduler":
        state_dict = torch.load(restore_pt)
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Argument `type` must be either `model` or `optimizer`.")

    logging.info(f"{type} weights restored from {restore_pt}.")
    return global_step


def save_pytorch_model(output_dir, model, file_name, max_save, type):
    """Save a trained model"""
    logging.info(f"** ** * Saving {type}: {file_name} ** ** * ")
    os.makedirs(output_dir, exist_ok=True)
    file_checkpoint = os.path.join(output_dir, type)
    if os.path.exists(file_checkpoint):
        with open(file_checkpoint, "r", encoding="utf-8") as r:
            file_lists = r.readlines()
    else:
        file_lists = []
    model_to_save = model.module if hasattr(model, "module") else model
    file_ckpt = os.path.join(output_dir, file_name)
    torch.save(model_to_save.state_dict(), file_ckpt)
    file_lists.append(file_ckpt + "\n")
    if len(file_lists) > max_save:
        for file in file_lists[:len(file_lists) - max_save]:
            try:
                os.remove(file.strip())
            except:
                logging.info(f"Failed to remove ckpt {file}.")
        open(file_checkpoint, "w").write("".join(file_lists[-max_save:]))
    else:
        open(file_checkpoint, "w").write("".join(file_lists))

def save(state, file):
    state_to_save = state.module if hasattr(state, "module") else state
    torch.save(state_to_save.state_dict(), file)

# def save_best_model(global_step, model, optimizer, acc, loss, hp):
def save_best_model(global_step, model, optimizer, scheduler, index_dict, core_index, hp):
    # index, core_index
    """
    保存最佳模型
    """
    os.makedirs(hp.dir_best_model, exist_ok=True)

    #sub_dir_best_model = os.path.join(hp.dir_best_model, str((global_step // hp.save_checkpoint_interval - 1) // 10))
    #os.makedirs(sub_dir_best_model, exist_ok=True)
    sub_dir_best_model = hp.dir_best_model
    file_evaluation_index = os.path.join(os.path.join(sub_dir_best_model, "evaluation_index.json"))
    if os.path.exists(file_evaluation_index):
        list_record = []
        for line in open(file_evaluation_index, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                list_record.append(json.loads(line))
        
        if len(list_record) < hp.max_num_best_model:
            record = {"global_step": global_step}
            record.update(index_dict)
            list_record.append(record)
            list_record = sorted(list_record, key=lambda x:x[core_index])
            with open(file_evaluation_index, "w", encoding="utf-8") as f:
                for line in list_record:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            
            # 保存模型
            save(model,   os.path.join(sub_dir_best_model, f"pytorch_model_{global_step}.pt"))
            # 保存优化器
            save(optimizer, os.path.join(sub_dir_best_model, f"pytorch_optimizer_{global_step}.pt"))
            # 保存scheduler
            save(scheduler, os.path.join(sub_dir_best_model, f"pytorch_scheduler_{global_step}.pt") )

            

        elif len(list_record) > 0 and list_record[0][core_index] < index_dict[core_index]: 
            record_min_acc = list_record[0]

            model_min_acc = os.path.join(sub_dir_best_model, f"pytorch_model_{record_min_acc['global_step']}.pt")
            optimizer_min_acc = os.path.join(sub_dir_best_model,
                                             f"pytorch_optimizer_{record_min_acc['global_step']}.pt")
            
            scheduler_min_acc = os.path.join(sub_dir_best_model,
                                             f"pytorch_scheduler_{record_min_acc['global_step']}.pt")

            if os.path.exists(model_min_acc):
                os.remove(model_min_acc)
            if os.path.exists(optimizer_min_acc):
                os.remove(optimizer_min_acc)
            if os.path.exists(scheduler_min_acc):
                os.remove(scheduler_min_acc)
            

            
            record = {"global_step": global_step}
            record.update(index_dict)
            list_record = list_record[1:] + [record]
            list_record = sorted(list_record, key=lambda x: x[core_index])
            with open(file_evaluation_index, "w", encoding="utf-8") as f:
                for line in list_record:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            
            # 保存模型
            save(model,   os.path.join(sub_dir_best_model, f"pytorch_model_{global_step}.pt"))
            # 保存优化器
            save(optimizer, os.path.join(sub_dir_best_model, f"pytorch_optimizer_{global_step}.pt"))
            # 保存scheduler
            save(scheduler, os.path.join(sub_dir_best_model, f"pytorch_scheduler_{global_step}.pt") )
            

    else:
        record = {"global_step": global_step}
        record.update(index_dict)
        list_record = [record]
        with open(file_evaluation_index, "w", encoding="utf-8") as f:
            for line in list_record:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        # 保存模型
        save(model,   os.path.join(sub_dir_best_model, f"pytorch_model_{global_step}.pt"))
        # 保存优化器
        save(optimizer, os.path.join(sub_dir_best_model, f"pytorch_optimizer_{global_step}.pt"))
        # 保存scheduler
        save(scheduler, os.path.join(sub_dir_best_model, f"pytorch_scheduler_{global_step}.pt") )
            