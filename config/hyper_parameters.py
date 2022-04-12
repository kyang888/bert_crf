import json

class HyperParams(object):
    def __init__(self,debug=True):
        self.task_name = "common_classification"
        self.cmd = "fine_tune"  # "train" | "eval"

        # dir
        self.dir_corpus = "/data/corpus"
        self.dir_training_data = f"../data"
        self.dir_checkpoint = f"../checkpoint"
        self.dir_summary = "../summary_finetune"

        # file
        self.file_vocab = "../resources/vocab.txt"
        self.file_ner_vocab = '../resources/ner_vocab.json'
        self.file_rela_vocab = '../resources/relation_vocab.json'

        # training data
        self.create_data_process_index = None
        self.corpus_type = "train"  # "train" | "dev" | "test"
        self.corpus_name_list = ["seven_corpus_shuffled"]

        self.do_lower_case = False
        self.do_random_next = False
        self.max_file_num = 50
        self.max_seq_length = 512
        self.short_seq_prob = 0.0
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = 76


        # model
        self.vocab_size = 21128
        self.hidden_size = 768
        #self.hidden_size = 1024
        self.intermediate_size = 3072
        #self.intermediate_size = 4096
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.num_attention_heads = 12
        #self.num_attention_heads = 16
        self.num_hidden_layers = 12
        #self.num_hidden_layers = 24
        self.initializer_range = 0.02
        self.hidden_act = "gelu"
        self.output_attentions = False

        self.output_hidden_states = False
        self.output_logit = False

        # train
        self.gradient_accumulation_steps = 1

        self.num_train_step = 1400000
        self.learning_rate = 5e-05
        self.weigth_decay_rate = 0.01
        self.warmup_steps = 1000
        self.optim_method = "RAdam"  # "RAdam" | "AdamWeightDecayOptimizer" | "Ranger"
        self.save_checkpoint_interval = 500
        self.map_location = "cpu"
        self.fp16 = False
        self.seed = 0
        self.file_dynamic_params = None

        # eval
        self.num_eval_step = 5000

        # fine tune
        self.num_labels = 29
        self.num_intention = 3
        self.dir_init_checkpoint = "../init_ckpt/pytorch_model_2701000.pt"
        self.file_prediction = "./prediction"

        # log
        self.checkpoint_max_keep = 10000

        # multi-process
        self.no_cuda = False
        self.local_rank = -1
        self.ranks_need_to_save = [0]

        # 增加
        self.use_checkpoint_sequential = False
        self.batch_size = 2
        self.ner_task = 'relation_common'
        self.num_classification_labels = 31
        self.num_relation_labels = 126
        self.ner_common_prob = 1.0
        self.ner_classification_prob = 0.0
        self.ner_relation_prob = 0.0

        self.num_eval_step_classification = 10000
        self.num_eval_step_common = 10000
        self.num_eval_step_relation = 10000


        # 加入词汇嵌入
        self.ues_pos_embeddings = True
        self.pos_vocab_size = 104
        # lstm config
        self.ust_lstm=False
        self.lstm_layers_num=1
        self.lstm_bidirectional=False


    @classmethod
    def init_from_parsed_args(cls, parsed_args):
        hp = cls()
        args_dict = vars(parsed_args)
        for k in args_dict:
            if args_dict[k] is not None:
                setattr(hp, k, args_dict[k])
        return hp

    @classmethod
    def init_from_json(cls, file_json):
        hp = cls()
        args_dict = dict(json.load(open(file_json, "r", encoding="utf-8")))
        for k in args_dict:
            if args_dict[k] is not None:
                setattr(hp, k, args_dict[k])
        return hp

    def set_horovod_para(self, local_rank=0, world_size=1):
        self.local_rank = local_rank
        self.world_size = world_size

    def to_json_string(self):
        return json.dumps(self.__dict__, indent=4)

        
