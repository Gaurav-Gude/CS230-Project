"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnlp.nn as nlp
import torchaudio as ta
import pickle


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

        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True, bidirectional=True)

        # the fully connected layer transforms the output to give the final output layer
        # self.attention = nlp.Attention(params.lstm_hidden_dim)
        # self.attention = Attention(params)

        # self.cosineSimilarity = nn.CosineSimilarity(dim=1)
        # self.mfcc = ta.transforms.MFCC(melkwargs={ 'n_fft' : 320 } )
        self.params = params

        self.fc = nn.Sequential(
            nn.Linear(params.lstm_hidden_dim * 2, 1),
            # nn.Linear(params.lstm_hidden_dim, params.vocab_size),
            # nn.ReLU(),
            # nn.Linear(300, params.embedding_dim)
        )
        self.crossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: Packed sequence

        Returns:
            out:

        Note: the dimensions after each step are provided
        """

        # run the LSTM along the sentences of length seq_len
        # dim:
        s, _ = self.lstm(s)

        # Unpack
        # s dim = batch_len, seq_len,  lstm_dim
        s, lengths = nn.utils.rnn.pad_packed_sequence(s, batch_first=True, total_length=self.params.max_seq_len)
        # make the Variable contiguous in memory (a PyTorch artefact)
        batch_size = s.shape[0]

        s = s.contiguous()
        # s dim = batch_len * seq_len ,lstm_dim
        s = s.view(-1, s.shape[2])

        # apply the Fully connectted Layer
        s = self.fc(s)  # dim: batch_len * seq_len ,lstm_dim, 1

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        s = F.sigmoid(s)  # dim: batch_size*seq_len x 1

        s = s.view(batch_size, -1)

        return s


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(params.max_mfcc_seq_length))
        nn.init.normal_(self.weights)

    def forward(self, input):
        """
        :param input:  dimentions batch size * max mfcc len * lstm dimentino
        :return:  batch size * lstm dimentinon
        """
        scores = F.softmax(self.weights)
        input = input.permute(0, 2, 1)  # dim batch size * lstm dimentino * max mfcc len
        scaled = torch.mul(input, scores)  # dim batch size *lstm dimentino * max mfcc len
        summed = torch.sum(scaled, dim=2)  # dim batch size  * lstm dimentino* 1
        return summed  # dim batch size  * lstm dimentino


def loss_fn(outputs, labels):
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

    # num_tokens = int(torch.sum(mask))
    # consSim = nn.CosineSimilarity(dim=1)
    mask = labels != -1
    loss = nn.BCELoss(reduction='none')(outputs, labels * mask)
    loss = loss * mask


    # Average of the
    return torch.sum(loss) / np.count_nonzero(mask)

    # compute cosine similarity for the whole batch
    # return torch.mean(1-consSim(outputs, labels))

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """
    outputs = outputs > .5
    label_max = np.argmax(labels, axis=1)
    count = 0
    for i in range(len(label_max)):
        if outputs[i][label_max[i]] :
            count+=1

    return count / len(label_max)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
