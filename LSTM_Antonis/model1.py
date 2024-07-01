import torch
import torch.nn as nn

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, dummy_dim, model_dim, num_heads, num_layers, seq_length,num_classes, dropout_rate=0.1,batch_first=True,norm_first=True,bias =False):
        super(TransformerTimeSeriesModel, self).__init__()
        
        self.input_dim = input_dim
        self.dummy_dim = dummy_dim
        self.model_dim = model_dim
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.dummy_embedding = nn.Linear(dummy_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dropout=dropout_rate)
        
        self.reconstruction_output = nn.Linear(model_dim, input_dim)
        self.classification_output = nn.Linear(model_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, dummy):
        # x shape: (batch_size, seq_length, input_dim)
        # dummy shape: (batch_size, seq_length, dummy_dim)
        batch_size, seq_length, _ = x.size()
        
        # Apply embedding and positional encoding to input features
        x = self.input_embedding(x)
        dummy = self.dummy_embedding(dummy.float())  # Ensure dummy is of type float
        
        # Combine embeddings and add positional encoding
        x_combined = x + dummy + self.positional_encoding

        #x_combined = x + dummy + self.positional_encoding[:, :seq_length, :]
        
        # Transformer expects input of shape (seq_length, batch_size, model_dim)
        x_combined = x_combined.permute(1, 0, 2)
        
        # Apply transformer
        memory = self.transformer(x_combined,x_combined)
        
        # Permute back to (batch_size, seq_length, model_dim)
        memory = memory.permute(1, 0, 2)
        
        # Apply dropout
        memory = self.dropout(memory)
        
        # Reconstruction output
        #recon_output = self.reconstruction_output(memory)
        
        # Classification output
        class_output = self.classification_output(memory)
        
        #return recon_output, class_output
        return class_output
