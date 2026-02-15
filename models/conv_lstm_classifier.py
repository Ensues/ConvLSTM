import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvLSTMCell(nn.Module):
    """
    The Single Memory Unit of the video.
    
    It looks at one frame of a video and updates its internal notepad (memory) 
    based on what it just saw and what it remembered from the frame before.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # Calculate padding to keep image size the same
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # The main neural network layer used inside this block
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # This function processes one step of time
        h_cur, c_cur = cur_state

        # Combine the new input with the previous memory
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Split the result into the 4 gates of an LSTM
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Calculate the new memory (c_next) and new output state (h_next)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        # Creates a blank slate (zeros) for the very first step
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    """
    The Observer of the video
    
    Watches the video frame by frame, and the Brain remembers what it has seen over time. 
    It combines Convolution (seeing patterns) with LSTM (remembering time).
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # Making sure the 'eyes' (kernel_size) are the right shape
        self._check_kernel_size_consistency(kernel_size)

        # Ensuring parameters are lists even if only one layer is provided
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Creating the actual LSTM layers stack
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # input_tensor format: [Batch, Time, Channel, Height, Width]
        
        if not self.batch_first:
            # (Time, Batch, Channel, Height, Width) -> (Batch, Time, Channel, Height, Width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Get Dimensions using the Correct Indices
        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Loop over TIME (seq_len), not Batch
            for t in range(seq_len):
                # Slice the time dimension: [Batch, Channel, H, W]
                # If layer_idx == 0, cur_layer_input is [B, T, C, H, W]
                # If layer_idx > 0, cur_layer_input is [B, T, Hidden, H, W] (from previous layer stack)
                
                input_t = cur_layer_input[:, t, :, :, :]
                
                h, c = self.cell_list[layer_idx](input_tensor=input_t, cur_state=[h, c])
                output_inner.append(h)

            # Stack along Time dimension (dim=1 because we enforce batch_first internally now)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        # Helper to create 'empty' memory (zeros) for the start of the video
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    # (Utility functions for checking data types and extending lists)
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
        
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMModel(nn.Module):
    """
    The Judge of the video.
    
    It uses the Observer to watch the video, then it takes the final 
    conclusion (the last frame's features) and maps it to a category (Class).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, height, width,
                 batch_first=True, bias=True, return_all_layers=False, num_classes=3, dropout_rate=0.5):
        super(ConvLSTMModel, self).__init__()
        
        # The core temporal processor
        # Ensure batch_first is passed correctly
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, 
                                 batch_first=batch_first, bias=bias, 
                                 return_all_layers=return_all_layers)
        
        # Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # The final decision layer, which converts features to classes (e.g., 0, 1, or 2)
        # Input to linear is: Hidden_Dim * H * W
        self.linear = nn.Linear(hidden_dim[-1] * height * width, num_classes)

    def forward(self, input_tensor, hidden_state=None):
        # Processes the whole video
        x, _ = self.convlstm(input_tensor)
        
        # x[0] shape is now guaranteed to be [Batch, Time, Hidden, H, W]
        # We take the last time step: x[0][:, -1, :, :, :]
        
        last_time_step = x[0][:, -1, :, :, :]
        
        # Flatten the image features into a long vector
        # Flatten starting from dimension 1 (Channels/Hidden)
        flattened = torch.flatten(last_time_step, start_dim=1)
        
        # Apply dropout for regularization (only active during training)
        flattened = self.dropout(flattened)
        
        # Predict the class
        output = self.linear(flattened)
        return output