"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        # self.lstm = nn.LSTM(params.speech_dim,
        #                     params.lstm_hidden_dim, batch_first=True, bidirectional=True)
        params.max_seq_len = int((params.max_mfcc_seq_length - params.kernel_size)/params.conv1_stride)
        self.conv1 = nn.Conv1d(params.speech_dim, params.speech_dim, params.kernel_size, stride=params.conv1_stride)

        self.lstm = nn.LSTM(params.speech_dim,
                            params.lstm_hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=params.dropout)


        # the fully connected layer transforms the output to give the final output layer
        # self.attention = nlp.Attention(params.lstm_hidden_dim)
        self.attention = Attention(params)

        #self.cosineSimilarity = nn.CosineSimilarity(dim=1)
        self.mfcc = ta.transforms.MFCC(melkwargs={ 'n_fft' : 320 } )
        self.params = params

        self.fc = nn.Sequential(
            nn.Linear(params.lstm_hidden_dim*2, params.outputHiddenDim),
            # nn.Linear(params.lstm_hidden_dim, params.vocab_size),
            nn.Dropout(p=params.dropout),
            nn.ReLU(),
            #nn.BatchNorm1d(params.outputHiddenDim),
            nn.Linear(params.outputHiddenDim, params.vocab_size),
        )

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: Packed sequence

        Returns:
            out:

        Note: the dimensions after each step are provided
        """
        s, lengths = nn.utils.rnn.pad_packed_sequence(s, batch_first=True, total_length=self.params.max_mfcc_seq_length)
        s = s.permute(0, 2, 1)
        s = self.conv1(s)
        lengths = (lengths - self.params.kernel_size) / self.params.conv1_stride
        s = s.permute(0, 2, 1)
        s = nn.utils.rnn.pack_padded_sequence(s, lengths, batch_first=True)
        # run the LSTM along the sentences of length seq_len
        # dim:
        s, _ = self.lstm(s)

        #Unpack
        s, lengths = nn.utils.rnn.pad_packed_sequence(s, batch_first=True, total_length=self.params.max_seq_len)

        # make the Variable contiguous in memory (a PyTorch artefact)
        #s = s.contiguous()

        # reshape the Variable so that each row contains one token
        # dim: batch_size*seq_len x lstm_hidden_dim
        s = self.attention(s)

        #Code ot get last of sequence
        # idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
        #     len(lengths), s.size(2))
        # idx = idx.unsqueeze(1)
        # if s.is_cuda:
        #     idx = idx.cuda(s.data.get_device())
        # s = s.gather(
        #     1, idx).squeeze(1)

        # apply the Fully connectted Layer
        s = self.fc(s)                   # dim: batch_size x embedding_dim

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        # return F.log_softmax(s, dim=1)   # dim: batch_size x vocab

        return s


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(params.max_seq_len))
        nn.init.normal_(self.weights)

    def forward(self, input):
        """
        :param input:  dimentions batch size * max mfcc len * lstm dimentino
        :return:  batch size * lstm dimentinon
        """
        scores = F.softmax(self.weights)
        input = input.permute(0,2,1) # dim batch size * lstm dimentino * max mfcc len
        scaled = torch.mul(input, scores) #dim batch size *lstm dimentino * max mfcc len
        summed = torch.sum(scaled, dim=2) #dim batch size  * lstm dimentino* 1
        return summed  #dim batch size  * lstm dimentino





def loss_fn(outputs, labels, params):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable)
        labels: (Variable)

    Returns:
        loss: (Variable)

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """


    #num_tokens = int(torch.sum(mask))
    # consSim = nn.CosineSimilarity(dim=1)
    return nn.CrossEntropyLoss().forward(outputs, labels)
    # compute cosine similarity for the whole batch
    # return torch.mean(1-consSim(outputs, labels))


def find_closest(embedding, word2Array):

    maxSim = 0
    candidate = None
    for key, value in word2Array.items() :
        sim = np.dot(value, embedding)/(np.linalg.norm(embedding)*np.linalg.norm(value))
        if sim > maxSim:
            maxSim = sim
            candidate = key
    return candidate




def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """
    outputs = torch.Tensor(outputs)
    outputs.requires_grad = False
    outputs = outputs.softmax(dim=1).argmax(dim=1).numpy()

    return np.count_nonzero(np.equal(outputs, labels))/float(labels.shape[0])


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
