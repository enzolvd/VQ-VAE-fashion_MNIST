import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(torch.nn.Module):
    """Define a Res connection block for encoder decoder"""
    def __init__(self, in_channel):        
        super().__init__()
        
        self.conv_block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels = in_channel,
                            out_channels = in_channel,
                            kernel_size=3,
                            stride=1,
                            padding='same'),
                    torch.nn.BatchNorm2d(in_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels = in_channel,
                            out_channels = in_channel,
                            kernel_size=3,
                            stride=1,
                            padding='same'),
                    torch.nn.BatchNorm2d(in_channel)                         
        )
        self.act = torch.nn.ReLU()

    def forward(self, input):
        out = self.conv_block(input)
        out += input 
        out = self.act(out)
        return out

class Encoder(torch.nn.Module):
    """Define Encoder block for VQ-VAE"""
    def __init__(self, embedding_dim, num_channel_first_conv=64, num_channel_second_conv=128):
        super().__init__()

        # First convolutionnal block Conv + Res Block 28x28x1 --> 28x28x64
        self.first_block = torch.nn.Sequential( 
                    torch.nn.Conv2d(in_channels=1,
                                    out_channels=num_channel_first_conv, 
                                    kernel_size=3,
                                    stride=1,
                                    padding='same'),
                    torch.nn.BatchNorm2d(num_channel_first_conv),
                    torch.nn.ReLU(),
                    ResBlock(num_channel_first_conv)
        )

        # Modified first downsampling block 28x28x64 --> 14x14x128
        self.first_down_sample = torch.nn.Sequential( 
            torch.nn.Conv2d(in_channels=num_channel_first_conv,
                            out_channels=num_channel_second_conv,
                            kernel_size=3,
                            stride=2,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(num_channel_second_conv),
            torch.nn.ReLU()
        )

        # Second convolutionnal block ResBlock 14x14x128 --> 14x14x128
        self.second_block = ResBlock(num_channel_second_conv)

        # Modified second downsampling block 14x14x128 --> 12x12xembedding_dim
        # Using a different kernel size and padding to achieve 12x12 output
        self.second_down_sample = torch.nn.Sequential( 
            torch.nn.Conv2d(in_channels=num_channel_second_conv,
                            out_channels=embedding_dim,
                            kernel_size=3,
                            stride=1,
                            padding=0
                            ),
            torch.nn.BatchNorm2d(embedding_dim),
            torch.nn.ReLU()
        )

        # Pre-quantization
        self.pre_quant = ResBlock(embedding_dim)

    def forward(self, input):
        first_block = self.first_block(input)
        first_down = self.first_down_sample(first_block)
        second_block = self.second_block(first_down)
        second_down = self.second_down_sample(second_block)
        out = self.pre_quant(second_down)
        return out

class Decoder(torch.nn.Module):
    """Define Decoder block for VQ-VAE"""
    def __init__(self, embedding_dim, num_channel_first_conv=64, num_channel_second_conv=128, num_channel_last_up=32):
        super().__init__()

        # Initial processing + first low resolution ResBlock 12x12xembedding_dim --> 12x12x128 --> 12x12x128
        self.init_process = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=embedding_dim,
                                out_channels=num_channel_second_conv,
                                kernel_size=3,
                                stride=1,
                                padding='same'),
                torch.nn.BatchNorm2d(num_channel_second_conv),
                torch.nn.ReLU(),
                ResBlock(num_channel_second_conv)
        )

        # Modified first upsampling process 12x12x128 --> 14x14x64
        self.first_upsampling = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=num_channel_second_conv,
                                         out_channels=num_channel_first_conv,
                                         kernel_size=3,
                                         stride=1,
                                         padding=0),
                torch.nn.BatchNorm2d(num_channel_first_conv),
                torch.nn.ReLU(),
                ResBlock(num_channel_first_conv)
        )

        # Second upsampling process 14x14x64 --> 28x28x32
        self.second_upsampling = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=num_channel_first_conv,
                                         out_channels=num_channel_last_up,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(num_channel_last_up),
                torch.nn.ReLU(),
                ResBlock(num_channel_last_up)
        )

        # Output block 28x28x32 --> 28x28x1
        self.output = torch.nn.Conv2d(in_channels=num_channel_last_up,
                                      out_channels=1,
                                      kernel_size=3,
                                      stride=1,
                                      padding='same'
                                      )
        
    def forward(self, input):
        processed = self.init_process(input)
        first_up = self.first_upsampling(processed)
        second_up = self.second_upsampling(first_up)
        out = self.output(second_up)
        return out


class VQ_VAE(nn.Module):
    """Define VQ-VAE, with Encoder, Decoder, and quantization"""
    def __init__(self, num_embedding, embedding_dim):
        # Define a VQ-VAE class with a codebook size of (num_embedding, embedding_dim)
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)
        self.embedding_vectors = nn.Embedding(num_embedding, embedding_dim)
        self.embedding_vectors.weight.data.uniform_(0, 2 / num_embedding)

    def forward(self, X):
        # Run the whole encoder-quantization-decoder steps

        #Encoder
        ze = self.encoder(X)

        #Quantization
        ze_flat = ze.permute(0,2,3,1).reshape(-1, self.embedding_dim)
        distances = torch.cdist(ze_flat, self.embedding_vectors.weight)
        best_indices = distances.argmin(dim=1)
        quantized_latent = self.embedding_vectors(best_indices)
        quantized_reshaped = quantized_latent.view(*ze.permute(0,2,3,1).shape)
        quantized_reshaped = quantized_reshaped.permute(0,3,1,2)        
        zq = ze + (quantized_reshaped - ze).detach()
        
        #Decoder
        reconstructed = self.decoder(zq)
        return reconstructed, ze, quantized_reshaped, best_indices

    def reconstruction_loss(self, input, output):
        return torch.nn.functional.mse_loss(input, output)
    
    def VQ_loss(self, ze, zq):
        return torch.nn.functional.mse_loss(ze.detach(), zq)
    
    def commitment_loss(self, ze, zq):
        return torch.nn.functional.mse_loss(ze, zq.detach())
    
    def loss(self, input, output, ze, zq, beta=0.25):
        reconstruction_loss = self.reconstruction_loss(input, output)
        VQ_loss = self.VQ_loss(ze, zq)
        commitment_loss = self.commitment_loss(ze, zq)
        global_loss = reconstruction_loss + VQ_loss + beta*commitment_loss
        return global_loss, reconstruction_loss, VQ_loss, commitment_loss