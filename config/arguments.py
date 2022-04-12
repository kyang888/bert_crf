import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--init_checkpoint", default="")
parser.add_argument("--dir_training_data", default=None)
parser.add_argument("--dir_raw_training_data", default=None)
parser.add_argument("--file_vocab", default="")
parser.add_argument("--dir_checkpoint", default=None)
parser.add_argument("--dir_summary", default=None)
parser.add_argument("--dir_log", default="", type=str)
parser.add_argument("--dir_best_model", default="", type=str)
parser.add_argument("--batch_size_train", default=None, type=int)
parser.add_argument("--batch_size_dev", default=None, type=int)
parser.add_argument("--num_train_step", default=None, type=int)
parser.add_argument("--learning_rate", default=None, type=float)
parser.add_argument("--gradient_accumulation_steps", default=None, type=float)
parser.add_argument("--use_checkpoint_sequential",action='store_true')
parser.add_argument("--chunks", default=12, type=int)
parser.add_argument("--warmup_step_rate", default=0, type=float)
parser.add_argument("--epoch", default=10, type=int)


parser.add_argument("--seed", default=123, type=int)
parser.add_argument("--save_checkpoint_interval", default=1000, type=int)
parser.add_argument("--interval_print_info", default=100, type=int)
parser.add_argument("--max_num_best_model", default=10, type=int)
parser.add_argument("--max_num_checkpoint", default=5, type=int)
parser.add_argument("--dir_init_checkpoint", default=None)
parser.add_argument("--do_lower_case", default=True, type=bool)
parser.add_argument("--max_seq_length", default=None, type=int)



# bert crf
parser.add_argument("--num_labels", default=None, type=int)
parser.add_argument("--lstm_layers_num", default=1, type=int)
parser.add_argument("--lstm_bidirectional",action='store_true')
parser.add_argument("--use_lstm",action='store_true')
parser.add_argument("--ner_file",default="", type=str)
parser.add_argument("--warmup_epoch", default=1.0, type=float)

if __name__ == '__main__':
    for k, v in vars(parser).items():
        print(k, v, type(v))

