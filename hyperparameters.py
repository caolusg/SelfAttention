class HyperParameters(object):
    def __init__(self):
        self.lr = 0.001
        self.epoches = 12
        self.batch = 16
        self.labelSize = 2
        self.embedding_num = 0
        self.embedding_dim = 300
        self.wordEmbeddingSize = 0
        self.unknown = None
        self.kernel_size = [1, 2, 3]
        self.kernel_num = 300
        self.word_Embedding = True
        self.word_Embedding_Path = "./data/converted_word_Subj.txt"
        self.pretrained_weight = None
        self.LSTM_hidden_dim = 50
        self.num_layers = 2
        self.class_num = 2
        self.dropout_embed = 0
        self.dropout = 0.2
        self.save_dir = "snapshot"
        self.snapshot = None
        self.LSTM_model = True
        self.att_size = 100