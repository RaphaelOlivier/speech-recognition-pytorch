import os
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.utils.data
import numpy as np
import datatools
import torch.nn.functional as F


class Baseline(torch.nn.Module):
    def __init__(self, nLabels, input_size=40, att_size=128, h_size_enc=256, h_size_dec=256, n_layers_dec=3, max_sentence=1000):
        super().__init__()
        self.encoderLSTM = torch.nn.LSTM(input_size=input_size, hidden_size=h_size_enc,
                                         num_layers=1, batch_first=False, bidirectional=True)
        self.encoderpLSTM1 = torch.nn.LSTM(input_size=4*h_size_enc, hidden_size=h_size_enc,
                                           num_layers=1, batch_first=False, bidirectional=True)
        self.encoderpLSTM2 = torch.nn.LSTM(input_size=4*h_size_enc, hidden_size=h_size_enc,
                                           num_layers=1, batch_first=False, bidirectional=True)
        self.encoderpLSTM3 = torch.nn.LSTM(input_size=4*h_size_enc, hidden_size=h_size_enc,
                                           num_layers=1, batch_first=False, bidirectional=True)

        self.decoder = torch.nn.LSTM(input_size=h_size_dec+2*att_size, hidden_size=h_size_dec,
                                     num_layers=n_layers_dec, batch_first=False, bidirectional=False)
        self.n_layers_dec = n_layers_dec
        self.h_size_dec = h_size_dec
        self.att_size = att_size
        self.keyProjection = torch.nn.Linear(2*h_size_enc, att_size)
        self.valueProjection = torch.nn.Linear(2*h_size_enc, att_size)
        self.queryProjection = torch.nn.Linear(h_size_dec, att_size)
        self.sf_att = torch.nn.Softmax(dim=0)
        self.characterProjection = torch.nn.Linear(h_size_dec+att_size, nLabels)
        #self.characterHidden = torch.nn.Linear(h_size_dec+att_size, h_size_dec)
        self.max_sentence = max_sentence
        self.act = torch.nn.LeakyReLU()
        self.input_size = input_size
        self.drop = torch.nn.Dropout(0.2)
        self.droplstm = LockedDropout(0.1)
        """
        for param in self.characterProjection.parameters():
            torch.nn.init.uniform(param, -0.1, 0.1)
        born = 1. / np.sqrt(h_size)
        for param in self.encoder.parameters():
            torch.nn.init.uniform(param, -born, born)
        for param in self.decoder.parameters():
            torch.nn.init.uniform(param, -born, born)
"""

    def embed_target(self, inp):
        return self.act(F.embedding(inp, self.characterProjection.weight))

    def forward(self, input_sequence, lengths=None, output_sequence=None, mode="train", n_preds=None):
        # input_sequence: Padded sequence
        if mode == "train":
            return self.decode_to_loss(input_sequence, lengths, output_sequence)
        elif mode == "eval":
            return self.decode_to_prediction(input_sequence)
        elif mode == "random":
            return self.random_search(input_sequence, n_preds=n_preds)

    def pool_and_encode(self, prev_o, lstm):
        i, lengths = pad_packed_sequence(prev_o, batch_first=True)
        i = self.droplstm(i)
        # print(i.size())
        if i.size(1) % 2 == 1:
            i = i[:, :-1, :]
            lengths = [x-1 for x in lengths]

        pooled = i.contiguous().view(i.size(0), i.size(1)//2, -1)
        # print(pooled.size())
        packed = pack_padded_sequence(pooled, [x//2 for x in lengths], batch_first=True)
        o, _ = lstm(packed)
        return o

    def encode(self, input_sequence, lengths):
        batch_size = input_sequence.size(1)
        # print(input_sequence.size())
        # print(lengths)
        packed = pack_padded_sequence(input_sequence, lengths, batch_first=False)
        # print(pad_packed_sequence(packed)[0].size())

        encoded, _ = self.encoderLSTM(packed)
        # print(pad_packed_sequence(encoded)[0].size())
        o1 = self.pool_and_encode(encoded, self.encoderpLSTM1)
        # print(pad_packed_sequence(o1)[0].data.size())
        o2 = self.pool_and_encode(o1, self.encoderpLSTM2)
        # print(pad_packed_sequence(o2)[0].size())
        o3 = self.pool_and_encode(o2, self.encoderpLSTM3)
        # print(pad_packed_sequence(o3)[0].size())

        output_encoder, new_lengths = pad_packed_sequence(o3, batch_first=False)
        output_encoder = self.droplstm(output_encoder)

        # print(new_lengths)
        keys = torch.cat([self.keyProjection(out).view(1, -1, self.att_size)
                          for out in output_encoder], dim=0)
        values = torch.cat([self.valueProjection(out).view(1, -1, self.att_size)
                            for out in output_encoder], dim=0)
        new_lengths = np.array(new_lengths)
        # print(lengths3)
        attention_mask = torch.Tensor((np.arange(keys.size()[0]).reshape(
            -1, 1) >= new_lengths.reshape((-1, len(new_lengths)))).astype(int)).cuda().byte().view(-1, len(new_lengths), 1)
        # print(attention_mask.cpu().numpy())
        # print(np.arange(keys.size()[0]).reshape(
        #     -1, 1))
        # print(lengths3.reshape((-1, len(lengths3))))
        # print(keys)
        return keys, values, attention_mask

    def attend(self, query, keys, values, attention_mask):
        # print(keys.size(), query.size())

        scores = torch.mul(keys, query.expand_as(keys)).sum(dim=2, keepdim=True)
        # print(scores)
        # print(scores.data.numpy())
        # print(attention_mask)
        scores[attention_mask] = float("-Inf")
        # print(scores.data.numpy())
        scores = self.sf_att(scores)

        # print(scores.data.numpy())

        context = torch.mul(scores.expand_as(values), values).sum(dim=0)
        # print(context.size())
        return context, l1_penalty(scores)

    def decode_to_loss(self, input_sequence, lengths, golden_output):
        # print(golden_output.size())
        ar = datatools.to_variable(torch.zeros(1))

        batch_size = golden_output.size(1)

        keys, values, attention_mask = self.encode(input_sequence, lengths)

        prev_context = datatools.to_variable(torch.zeros(1, batch_size, self.att_size))
        prev_state = datatools.to_variable(torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec)), datatools.to_variable(
            torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec))
        decoder_inputs = self.embed_target(golden_output)
        outputs_decoder = []
        for i in range(decoder_inputs.size(0)):
            target_input = decoder_inputs[i:i+1, :, :]
            # print(target_input.size(), prev_context.size())
            inp = torch.cat([target_input, prev_context], dim=2)
            decoded, new_state = self.decoder(inp, prev_state)
            query = self.queryProjection(self.drop(decoded))
            new_context, lasso = self.attend(query, keys, values, attention_mask)
            # print(new_state[0][-1].size(), new_context.size())
            output = self.characterProjection(
                self.drop(torch.cat([new_state[0][-1], new_context], dim=1)))
            outputs_decoder.append(output.view(1, output.size(0), -1))
            prev_state, prev_context = new_state, new_context.view(1, new_context.size(0), -1)

        outputs = torch.cat(outputs_decoder)
        ar += l2_penalty(outputs) + lasso

        # print(probs.size())
        return outputs, ar

    def decode_to_prediction(self, input_sequence, vocab=None):
        # only one sentence
        input_sequence = input_sequence.view(input_sequence.size(0), 1, input_sequence.size(1))
        lengths = [input_sequence.size(0)]

        batch_size = 1

        keys, values, attention_mask = self.encode(input_sequence, lengths)

        prev_context = datatools.to_variable(torch.zeros(1, batch_size, self.att_size))
        prev_state = datatools.to_variable(torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec)), datatools.to_variable(
            torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec))

        prev_char = 0
        characters = [0]
        new_char = None
        while new_char != 0:
            # print(torch.Tensor([[prev_char]]).long().size())
            target_input = self.embed_target(datatools.to_variable(
                torch.Tensor([[prev_char]]).long()))
            # print(target_input.size(), prev_context.size())
            inp = torch.cat([target_input, prev_context], dim=2)
            decoded, new_state = self.decoder(inp, prev_state)
            query = self.queryProjection(decoded)
            new_context, lasso = self.attend(query, keys, values, attention_mask)
            output = self.characterProjection(torch.cat([new_state[0][-1], new_context], dim=1))
            # print(output.size())
            _, new_char = torch.max(output, dim=1)
            new_char = new_char.data.cpu().numpy()[0].item()
            if(len(characters) > self.max_sentence):
                new_char = 0
            characters.append(new_char)
            prev_char = new_char
            prev_state, prev_context = new_state, new_context.view(1, new_context.size(0), -1)
        # print(probs.size())
        return characters

    def random_search(self, input_sequence, vocab=None, n_preds=100):
        # only one sentence
        print("new sentence")
        sf = torch.nn.Softmax(dim=1)
        prev_characters = []
        prev_logprob = datatools.to_variable(torch.Tensor([float("-Inf")]))

        input_sequence = input_sequence.view(input_sequence.size(0), 1, input_sequence.size(1))
        lengths = [input_sequence.size(0)]

        batch_size = 1

        keys, values, attention_mask = self.encode(input_sequence, lengths)

        for i in range(n_preds):
            total_logprob = 0

            prev_context = datatools.to_variable(torch.zeros(1, batch_size, self.att_size))
            prev_state = datatools.to_variable(torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec)), datatools.to_variable(
                torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec))

            prev_char = 0
            characters = [0]
            new_char = None
            while new_char != 0:
                # print(torch.Tensor([[prev_char]]).long().size())
                target_input = self.embed_target(datatools.to_variable(
                    torch.Tensor([[prev_char]]).long()))
                # print(target_input.size(), prev_context.size())
                inp = torch.cat([target_input, prev_context], dim=2)
                decoded, new_state = self.decoder(inp, prev_state)
                query = self.queryProjection(decoded)
                new_context, lasso = self.attend(query, keys, values, attention_mask)
                output = self.characterProjection(torch.cat([new_state[0][-1], new_context], dim=1))
                probs = sf(output)
                # print(output.size())
                sample = torch.rand(1)
                new_char = -1
                acc = datatools.to_variable(torch.zeros(1))
                # print(acc.size())
                # print(probs.size())
                while(acc.data[0] < sample[0]):
                    new_char += 1
                    acc += probs[0, new_char]

                # new_char = new_char.data.cpu().numpy()[0].item()
                if(len(characters) > self.max_sentence):
                    new_char = 0
                    total_logprob -= 1000
                characters.append(new_char)
                total_logprob += torch.log(probs[0, new_char])
                prev_char = new_char
                prev_state, prev_context = new_state, new_context.view(1, new_context.size(0), -1)
            total_logprob /= len(characters)
            if(total_logprob.data[0] > prev_logprob.data[0]):
                prev_logprob = total_logprob
                prev_characters = characters
                print("for search", i, ": prob", prev_logprob.data[0])
                # print(prev_characters)
        # print(probs.size())
        prev_context = datatools.to_variable(torch.zeros(1, batch_size, self.att_size))
        prev_state = datatools.to_variable(torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec)), datatools.to_variable(
            torch.zeros(self.n_layers_dec, batch_size, self.h_size_dec))

        prev_char = 0
        characters = [0]
        new_char = None
        total_logprob = 0
        while new_char != 0:
            # print(torch.Tensor([[prev_char]]).long().size())
            target_input = self.embed_target(datatools.to_variable(
                torch.Tensor([[prev_char]]).long()))
            # print(target_input.size(), prev_context.size())
            inp = torch.cat([target_input, prev_context], dim=2)
            decoded, new_state = self.decoder(inp, prev_state)
            query = self.queryProjection(decoded)
            new_context, lasso = self.attend(query, keys, values, attention_mask)
            output = self.characterProjection(torch.cat([new_state[0][-1], new_context], dim=1))
            probs = sf(output)
            # print(output.size())
            _, new_char = torch.max(output, dim=1)
            new_char = new_char.data.cpu().numpy()[0].item()
            if(len(characters) > self.max_sentence):
                new_char = 0
                total_logprob -= 1000

            characters.append(new_char)
            prev_char = new_char
            prev_state, prev_context = new_state, new_context.view(1, new_context.size(0), -1)
            total_logprob += torch.log(probs[0, new_char])
        total_logprob /= len(characters)
        print("for greedy search : prob", total_logprob.data[0])
        if(total_logprob.data[0] > prev_logprob.data[0]):
            prev_logprob = total_logprob
            prev_characters = characters
        return prev_characters


def l1_penalty(var):
    return torch.abs(var).sum()


def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


class LockedDropout(torch.nn.Module):
    def __init__(self, p=None):
        super(LockedDropout, self).__init__()
        self.dropout = p

    def forward(self, x):
        if not self.training or not self.dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x
