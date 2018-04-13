import os
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.utils.data
import numpy as np
import datatools
import torch.nn.functional as F


class ToySeq2Seq(torch.nn.Module):
    def __init__(self, input_size, h_size, n_layers, nLabels, max_sentence=100):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_size=input_size, hidden_size=h_size,
                                     num_layers=n_layers, batch_first=False, bidirectional=False)
        self.decoder = torch.nn.LSTM(input_size=h_size, hidden_size=h_size,
                                     num_layers=n_layers, batch_first=True, bidirectional=False)

        self.characterProjection = torch.nn.Linear(h_size, nLabels)
        self.max_sentence = max_sentence
        self.act = torch.nn.Softplus()
        self.input_size = input_size
        for param in self.characterProjection.parameters():
            torch.nn.init.uniform(param, -0.1, 0.1)
        born = 1. / np.sqrt(h_size)
        for param in self.encoder.parameters():
            torch.nn.init.uniform(param, -born, born)
        for param in self.decoder.parameters():
            torch.nn.init.uniform(param, -born, born)

    def embed_target(self, inp):
        return self.act(F.embedding(inp, self.characterProjection.weight))

    def forward(self, input_sequence, lengths=None, output_sequence=None, mode="train"):
        # input_sequence: Padded sequence
        if mode == "train":
            return self.decode_to_loss(input_sequence, lengths, output_sequence)
        else:
            return self.decode_to_prediction(input_sequence)

    def decode_to_loss(self, input_sequence, lengths, golden_output):
        packed = pack_padded_sequence(input_sequence, lengths, batch_first=True)
        _, state = self.encoder(packed)
        # print(state[0].size())
        ar = Variable(torch.zeros(1).cuda())
        # print(unpacked.size())
        decoder_input = self.embed_target(golden_output)
        # print(decoder_input.size())
        decoded, _ = self.decoder(decoder_input, state)

        outputs = self.characterProjection(decoded)
        # print(probs.size())
        return outputs, ar

    def decode_to_prediction(self, input_sequence):
        packed = input_sequence.view(-1, 1, self.input_size)
        _, state = self.encoder(packed)
        w = Variable(torch.zeros(1)).long().view(1, 1)
        l = [0]

        # first iteration
        inp = self.embed_target(w)
        h, state = self.decoder(inp, state)
        out = self.characterProjection(h)
        _, ind = torch.max(out, dim=2)
        word = ind.data.numpy()[0, 0]
        l.append(word)
        print(word)
        while(word != 0):

            inp = self.embed_target(w)
            h, state = self.decoder(inp, state)
            out = self.characterProjection(h)
            # print(out)
            _, ind = torch.max(out, dim=2)
            word = ind.data.numpy()[0, 0]
            if(len(l) > 100):
                word = 0
            l.append(word)
            # print(word)
        return np.array(l)


class Baseline(torch.nn.Module):
    def __init__(self, nLabels, input_size=40, att_size=128, h_size_enc=256, h_size_dec=256, max_sentence=1000):
        super().__init__()
        self.encoderLSTM = torch.nn.LSTM(input_size=input_size, hidden_size=h_size_enc,
                                         num_layers=1, batch_first=False, bidirectional=True)
        self.encoderpLSTM1 = torch.nn.LSTM(input_size=4*h_size_enc, hidden_size=h_size_enc,
                                           num_layers=1, batch_first=False, bidirectional=True)
        self.encoderpLSTM2 = torch.nn.LSTM(input_size=4*h_size_enc, hidden_size=h_size_enc,
                                           num_layers=1, batch_first=False, bidirectional=True)
        self.encoderpLSTM3 = torch.nn.LSTM(input_size=4*h_size_enc, hidden_size=h_size_enc,
                                           num_layers=1, batch_first=False, bidirectional=True)

        self.decoder_cell1 = torch.nn.LSTMCell(
            input_size=h_size_dec+att_size, hidden_size=h_size_dec)
        self.decoder_cell2 = torch.nn.LSTMCell(
            input_size=h_size_dec, hidden_size=h_size_dec)
        self.decoder_cell3 = torch.nn.LSTMCell(
            input_size=h_size_dec, hidden_size=h_size_dec)

        self.h_size_dec = h_size_dec
        self.att_size = att_size
        self.keyProjection = torch.nn.Linear(2*h_size_enc, att_size)
        self.valueProjection = torch.nn.Linear(2*h_size_enc, att_size)
        self.queryProjection = torch.nn.Linear(h_size_dec, att_size)
        self.sf_att = torch.nn.Softmax(dim=0)
        self.characterProjection = torch.nn.Linear(h_size_dec, nLabels)
        self.characterHidden = torch.nn.Linear(h_size_dec+att_size, h_size_dec)
        self.max_sentence = max_sentence
        self.act = torch.nn.Softplus()
        self.input_size = input_size

        self.initial_hidden_state = torch.nn.Parameter(torch.zeros(3, 1, h_size_dec).cuda())
        self.initial_cell_state = torch.nn.Parameter(torch.zeros(3, 1, h_size_dec).cuda())
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

    def forward(self, input_sequence, lengths=None, output_sequence=None, mode="train"):
        # input_sequence: Padded sequence
        if mode == "train":
            return self.decode_to_loss(input_sequence, lengths, output_sequence)
        else:
            return self.decode_to_prediction(input_sequence)

    def pool_and_encode(self, prev_o, lstm):
        i, lengths = pad_packed_sequence(prev_o, batch_first=True)
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
        #print(keys.size(), query.size())

        scores = torch.mul(keys, query.expand_as(keys)).sum(dim=2, keepdim=True)
        # print(scores)
        # print(scores.data.numpy())
        # print(attention_mask)
        scores[attention_mask] = float("-Inf")
        # print(scores.data.cpu().numpy())
        scores = self.sf_att(scores)

        # print(scores)
        #print(scores.size(), values.size())
        context = torch.mul(scores.expand_as(values), values).sum(dim=0)
        # print(context.size())
        return context

    def decoder(self, inp, prev_state):

        h1 = self.decoder_cell1(inp, prev_state[0])
        h2 = self.decoder_cell2(h1[0], prev_state[1])
        h3 = self.decoder_cell3(h2[0], prev_state[2])

        return [h1, h2, h3]

    def decode_to_loss(self, input_sequence, lengths, golden_output):
        # print(golden_output)
        batch_size = golden_output.size(1)
        # print(self.initial_hidden_state)
        keys, values, attention_mask = self.encode(input_sequence, lengths)

        prev_state = [(h.expand(batch_size, self.h_size_dec), c.expand(batch_size, self.h_size_dec))
                      for (h, c) in zip(self.initial_hidden_state, self.initial_cell_state)]
        query = self.queryProjection(prev_state[2][0])
        prev_context = self.attend(query, keys, values, attention_mask)

        decoder_inputs = self.embed_target(golden_output)
        # print(decoder_inputs.size())
        outputs_decoder = []
        for i in range(decoder_inputs.size(0)):
            target_input = decoder_inputs[i, :, :]

            # print(target_input.size(), prev_context.size())
            inp = torch.cat([target_input, prev_context], dim=1)

            new_state = self.decoder(inp, prev_state)
            query = self.queryProjection(new_state[-1][0])
            new_context = self.attend(query, keys, values, attention_mask)
            # print(new_state[0][-1].size(), new_context.size())
            output = self.characterProjection(self.act(self.characterHidden(
                torch.cat([new_state[-1][0], new_context], dim=1))))
            outputs_decoder.append(output.view(1, output.size(0), -1))
            prev_state, prev_context = new_state, new_context

        outputs = torch.cat(outputs_decoder)
        # print(probs.size())
        return outputs

    def decode_to_prediction(self, input_sequence, vocab=None):
        # only one sentence
        input_sequence = input_sequence.view(input_sequence.size(0), 1, input_sequence.size(1))
        lengths = [input_sequence.size(0)]

        batch_size = 1

        keys, values, attention_mask = self.encode(input_sequence, lengths)

        prev_state = [(h.expand(batch_size, self.h_size_dec), c.expand(batch_size, self.h_size_dec))
                      for (h, c) in zip(self.initial_hidden_state, self.initial_cell_state)]
        query = self.queryProjection(prev_state[2][0])
        prev_context = self.attend(query, keys, values, attention_mask)
        prev_char = 0
        characters = [0]
        new_char = None
        while new_char != 0:
            # print(torch.Tensor([[prev_char]]).long().size())
            target_input = self.embed_target(datatools.to_variable(
                torch.Tensor([prev_char]).long()))
            # print(target_input.size(), prev_context.size())
            inp = torch.cat([target_input, prev_context], dim=1)
            new_state = self.decoder(inp, prev_state)
            query = self.queryProjection(new_state[2][0])
            new_context = self.attend(query, keys, values, attention_mask)
            output = self.characterProjection(self.act(self.characterHidden(
                torch.cat([new_state[2][0], new_context], dim=1))))
            # print(output.size())

            _, new_char = torch.max(output, dim=1)
            new_char = new_char.data.cpu().numpy()[0].item()
            if(len(characters) > self.max_sentence):
                new_char = 0
            characters.append(new_char)
            prev_char = new_char
            prev_state, prev_context = new_state, new_context

        return characters
