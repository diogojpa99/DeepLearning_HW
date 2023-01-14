import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):

        super(Attention, self).__init__()
        "Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)"
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        query,
        encoder_outputs,
        src_lengths,
    ):
        # query: (batch_size, 1, hidden_dim)
        # encoder_outputs: (batch_size, max_src_len, hidden_dim)
        # src_lengths: (batch_size)

        # Create the mask for the padding tokens
        src_seq_mask = ~self.sequence_mask(src_lengths)
        
        #print(src_seq_mask)  

        # Linear layer that computes the query
        q = self.linear_in(query)
        # linear layer that computes the context vector
        #c = self.linear_in(encoder_outputs)

        # Compute the attention scores
        attn_scores = torch.bmm(q, encoder_outputs.transpose(1, 2))

        # Apply mask to the attention scores
        attn_scores.masked_fill_(src_seq_mask.unsqueeze(1), float("-inf"))

        # Compute the attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute the context vector
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # Concatenate the query and context vector
        concat_query_context = torch.cat((query, context), dim=-1)

        # Linear layer that computes the attention output
        attn_out = torch.tanh(self.linear_out(concat_query_context))

        return attn_out


    # def forward(
    #     self,
    #     query,
    #     encoder_outputs,
    #     src_lengths,
    # ):
    #     # query: (batch_size, 1, hidden_dim)
    #     # encoder_outputs: (batch_size, max_src_len, hidden_dim)
    #     # src_lengths: (batch_size)
    #     # we will need to use this mask to assign float("-inf") in the attention scores
    #     # of the padding tokens (such that the output of the softmax is 0 in those positions)
    #     # Tip: use torch.masked_fill to do this
    #     # src_seq_mask: (batch_size, max_src_len)
    #     # the "~" is the elementwise NOT operator
    #     src_seq_mask = ~self.sequence_mask(src_lengths)
    #     #############################################
    #     # TODO: Implement the forward pass of the attention layer
    #     # Hints:
    #     # - Use torch.bmm to do the batch matrix multiplication
    #     #    (it does matrix multiplication for each sample in the batch)
    #     # - Use torch.softmax to do the softmax
    #     # - Use torch.tanh to do the tanh
    #     # - Use torch.masked_fill to do the masking of the padding tokens
    #     #############################################
    #     raise NotImplementedError
    #     #############################################
    #     # END OF YOUR CODE
    #     #############################################
    #     # attn_out: (batch_size, 1, hidden_size)
    #     # TODO: Uncomment the following line when you implement the forward pass
    #     # return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        # print('src', src[0])
        # print('src', src.shape)
        embeddings = self.embedding(src)
        # print('embeddings', embeddings.shape)
        if self.training:
            embeddings = self.dropout(embeddings)


        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, final_hidden = self.lstm(packed_input)

        encoder_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        if self.training:
            encoder_output = self.dropout(encoder_output)

        # Return the encoder output and the final hidden state
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        return encoder_output, final_hidden





class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)
        
        embeddings = self.embedding(tgt)
        if self.training:
            embeddings = self.dropout(embeddings)
        outputs, dec_state = self.lstm(embeddings, dec_state)
        if self.training:
            outputs = self.dropout(outputs)
            outputs = outputs[:, :-1, :]

        if self.attn is not None:
            outputs = self.attn(outputs, encoder_outputs, src_lengths)


        

        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
        # TODO: Uncomment the following line when you implement the forward pass
        return outputs, dec_state

    

    # def forward(
    #     self,
    #     tgt,
    #     dec_state,
    #     encoder_outputs,
    #     src_lengths,
    # ):
    #     # tgt: (batch_size, max_tgt_len)
    #     # dec_state: tuple with 2 tensors
    #     # each tensor is (num_layers * num_directions, batch_size, hidden_size)
    #     # encoder_outputs: (batch_size, max_src_len, hidden_size)
    #     # src_lengths: (batch_size)

    #     # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
    #     # if they are of size (num_layers*num_directions, batch_size, hidden_size)

    #     # print('tgt', tgt.shape)
    #     # print('tgt[0]', tgt[0])
    #     # print('tgt[0]', tgt[5])
    #     # print('tgt[0]', tgt[50])
    #     # print('dec0', dec_state[0].shape)
    #     # print('dec1', dec_state[1].shape)
    #     if dec_state[0].shape[0] == 2:
    #         dec_state = reshape_state(dec_state)
    #     # print('dec0', dec_state[0].shape)
    #     # print('dec1', dec_state[1].shape)
    #     # print('scr', src_lengths.shape)
    #     # print(encoder_outputs.shape)

    #     # Initialize an empty list to store the decoder outputs
    #     outputs = []

    #     # Initialize the current input to the decoder as the start-of-sequence token
    #     # curr_input = tgt[:, 0].unsqueeze(1)
    #     # print('tgt[:, 0]', tgt[:, 0].shape)
    #     # print('curr_input', curr_input.shape)

    #     # Loop through the target sequence one token at a time
    #     for t in range(tgt.shape[1]):
    #         curr_input = tgt[:, t].unsqueeze(1)
    #         # Embed the current input token
    #         embedded = self.embedding(curr_input)

    #         # Apply dropout to the embedded input
    #         embedded = self.dropout(embedded)

    #         # Forward pass through the LSTM
    #         output, dec_state = self.lstm(embedded, dec_state)

    #         # If the attention mechanism is implemented, apply attention to the output
    #         # if self.attn is not None:
    #         #     output = self.attn(
    #         #         output,
    #         #         encoder_outputs,
    #         #         src_lengths,
    #         #     )

    #         # Append the output to the list of decoder outputs
    #         outputs.append(output)

    #         # Update the current input to the decoder to the next target token
    #         curr_input = tgt[:, t].unsqueeze(1)
        
    #     # Concatenate the decoder outputs to form the final output sequence
    #     outputs = torch.cat(outputs, dim=1)

    #     # Return the decoder outputs and the final decoder state
    #     return outputs, dec_state


    # def forward(
    #     self,
    #     tgt,
    #     dec_state,
    #     encoder_outputs,
    #     src_lengths,
    # ):
    #     # tgt: (batch_size, max_tgt_len)
    #     # dec_state: tuple with 2 tensors
    #     # each tensor is (num_layers * num_directions, batch_size, hidden_size)
    #     # encoder_outputs: (batch_size, max_src_len, hidden_size)
    #     # src_lengths: (batch_size)
    #     # bidirectional encoder outputs are concatenated, so we may need to
    #     # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
    #     # if they are of size (num_layers*num_directions, batch_size, hidden_size)
    #     if dec_state[0].shape[0] == 2:
    #         dec_state = reshape_state(dec_state)

    #     #############################################
    #     # TODO: Implement the forward pass of the decoder
    #     # Hints:
    #     # - the input to the decoder is the previous target token,
    #     #   and the output is the next target token
    #     # - New token representations should be generated one at a time, given
    #     #   the previous token representation and the previous decoder state
    #     # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
    #     # if self.attn is not None:
    #     #     output = self.attn(
    #     #         output,
    #     #         encoder_outputs,
    #     #         src_lengths,
    #     #     )
        
    #     # Initialize the outputs tensor to store the decoder output at each time step
    #     print(encoder_outputs)

    #     outputs = torch.zeros(tgt.size(0), tgt.size(1), self.hidden_size)

    #     # Iterate over the time steps of the target sequences
    #     for t in range(tgt.size(1)):
    #         # Generate the new token representation given the previous token representation
    #         # and the previous decoder state
    #         output, dec_state = self.lstm(tgt[:, t], dec_state)

    #         # # If the attention mechanism is enabled, apply it to the output
    #         # if self.attn is not None:
    #         #     output = self.attn(
    #         #         output,
    #         #         encoder_outputs,
    #         #         src_lengths,
    #         #     )

    #         # Store the output for this time step in the outputs tensor
    #         outputs[:, t, :] = output

    #     # Return the final outputs tensor and the final decoder state
    #     return outputs, dec_state
    
    # def forward(
    #     self,
    #     tgt,
    #     dec_state,
    #     encoder_outputs,
    #     src_lengths,
    # ):
    #     # tgt: (batch_size, max_tgt_len)
    #     # dec_state: tuple with 2 tensors
    #     # each tensor is (num_layers * num_directions, batch_size, hidden_size)
    #     # encoder_outputs: (batch_size, max_src_len, hidden_size)
    #     # src_lengths: (batch_size)

    #     # Reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
    #     # if they are of size (num_layers*num_directions, batch_size, hidden_size)
    #     if dec_state[0].shape[0] == 2:
    #         dec_state = reshape_state(dec_state)

    #     # Initialize an empty list to store the output sequences
    #     outputs = []

    #     # Embed the target sequences
    #     embedded = self.embedding(tgt)

    #     # Loop through each time step in the target sequences
    #     for t in range(tgt.size(1)):
    #         # Select the current time step input
    #         input = embedded[:, t, :].unsqueeze(1)

    #         # Forward pass through the LSTM
    #         output, dec_state = self.lstm(input, dec_state)

    #         # Compute the attention weights
    #         # attn_weights = self.attn(output, encoder_outputs, src_lengths)

    #         # Compute the context vector
    #         # context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

    #         # Concatenate the output and the context vector
    #         # output = torch.cat((output, context), dim=2)

    #         # Apply dropout to the output
    #         output = self.dropout(output)

    #         # Append the output to the list of outputs
    #         outputs.append(output)

    #     # Stack the outputs into a single tensor
    #     outputs = torch.cat(outputs, dim=1)

    #     # Return the outputs and the final decoder state
    #     return outputs, dec_state




class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
