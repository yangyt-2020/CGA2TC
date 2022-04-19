

class CONFIG(object):
    """docstring for CONFIG"""
    def __init__(self):
        super(CONFIG, self).__init__()
        
        self.dataset = 'ohsumed'
        self.model = 'gcn'  # 'gcn', 'gcn_cheby', 'dense'
        self.learning_rate = 0.02 # Initial learning rate.0.02#HGAT 0.005
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.CL_weight = 1#0.0001
        self.param = 'uniform'     
        self.sample_type = 0 #sample
        self.sample_size = 0.8
        self.encoder_type = 1#
        self.hidden1 = 512 # Number of units in hidden layer 1.
        self.pre_epoch = 40
 
        self.threshold = 0.7
        self.second_hidn = 200 #
        self.weight_decay = 5e-08   # Weight for L2 loss on embedding matrix.#5e-8
        self.early_stopping = 10 # Tolerance for early stopping (# of epochs).
        self.max_degree = 3      # Maximum Chebyshev polynomial degree.
        self.epochs  = 200 # Number of epochs to train.# for mr epoch=60
        self.time = 10   #run time 
