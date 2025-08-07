import torch
import torchvision
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torchinfo import summary
from tqdm.auto import tqdm
import cv2
import pathlib
from enum import Enum
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,Dataset
import re
import base64
import numpy as np


transform_videos_prueba=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

transform_none=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

transform_undo_resize=transforms.Compose([
    transforms.Resize((1080,1920))
])

def get_frames(video_path,n_frames,transform=transform_videos_prueba):
        cap = cv2.VideoCapture(video_path)

        video_length=cap.get(cv2.CAP_PROP_FRAME_COUNT)

        frames=[]

        ret,frame=None,None

        frame_step=int(video_length//n_frames)
        
        for _ in range(n_frames):
            for _ in range(frame_step):
                ret,frame=cap.read()
                # frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if ret:
                if transform:
                    frame=transform(frame)
                    frames.append(frame)
                else:
                    frame=transform_none(frame)
                    frames.append(frame)

        cap.release()
        video_tensor=torch.stack(frames)

        return video_tensor

device='cuda' if torch.cuda.is_available() else 'cpu'
device

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels:int=3,
                patch_size:int=16,
                embedding_dim:int=768):
        super().__init__()

        self.patcher=nn.Conv2d(in_channels=in_channels,
                            out_channels=embedding_dim,
                            kernel_size=patch_size,
                            stride=patch_size,
                            padding=0)
        self.flatten=nn.Flatten(start_dim=2,end_dim=3) #only flatten the feature map dimension

        self.patch_size=patch_size
    
    def forward(self,x:torch.tensor):
        #image_resolution=x.shape[-1] #quadratic images
        bs,frames,channels,height,width=x.shape
        #print(f"Image resolution {image_resolution}")
        assert height % self.patch_size == 0, f"Input image size must be divisible by patch size"

        x=x.view(bs*frames,channels,height,width)
        
        x_patched=self.patcher(x)

        #print(f"Shape after Patch Embedding {x_patched.shape}")

        x_flattened=self.flatten(x_patched)

        #print(f"Shape after flatten {x_flattened.shape}")

        x=x_flattened.permute(0,2,1)

        _,n_patches,emb_dim=x.shape

        x=x.view(bs,frames,n_patches,emb_dim)

        return x

class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels:int=3,
                tubelet_size:tuple=(2,16,16),
                d_model:int=768):
        super().__init__()

        self.patcher=nn.Conv3d(in_channels=in_channels,
                            out_channels=d_model,
                            kernel_size=tubelet_size,
                            stride=tubelet_size)
        
        self.flatten=nn.Flatten(start_dim=3,end_dim=4)
        
    def forward(self,x:torch.tensor):
        x=x.permute(0,2,1,3,4)
        x_patched=self.patcher(x)
        #print(f"Shape after the pacth embedding: {x_patched.shape}")
        x_flatten=self.flatten(x_patched)
        #print(f"Shape after the flatten: {x_flatten.shape}")
        x=x_flatten.permute(0,2,3,1)
        #print(f"Shape after the permute: {x.shape}")
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model,
                num_heads,
                num_layers,
                mlp_size:int,
                activaction:str='gelu',
                batch_first:bool=True,
                norm_first:bool=True,
                dropout:int=0.1):
        super().__init__()
        self.transformer_encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                                nhead=num_heads,
                                                                dim_feedforward=mlp_size,
                                                                dropout=dropout,
                                                                activation=activaction,
                                                                batch_first=batch_first,
                                                                norm_first=norm_first)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                    num_layers=num_layers)
    
    def forward(self,x:torch.tensor):
        x=self.transformer_encoder(x)
        return x

class ViViTFactorized(nn.Module):
    def __init__(self, in_channels:int=3,
                num_frames:int=8,
                img_size:int=224,
                tubelet_size:tuple=(2,16,16),
                d_model:int=768,
                num_heads:int=8,
                num_layers:int=6,
                mlp_size:int=3072,
                num_classes:int=3):
        super().__init__()

        # self.patch_embedding=PatchEmbedding(in_channels=in_channels,
        #                                     patch_size=patch_size,
        #                                     embedding_dim=d_model)

        self.patch_embedding=TubeletEmbedding(in_channels=in_channels,
                                            tubelet_size=tubelet_size,
                                            d_model=d_model)

        num_patches=(img_size // tubelet_size[1]) ** 2

        #cls tokens para el espacio y tiempo

        self.spatial_cls_token=nn.Parameter(torch.randn(1,1,d_model))
        
        self.temporal_cls_token=nn.Parameter(torch.randn(1,1,d_model))

        #Posicionales separadas para espacio y tiempo

        self.spatial_pos_embedding=nn.Parameter(torch.randn(1,num_patches+1,d_model))

        self.temporal_pos_embedding=nn.Parameter(torch.randn(1,num_frames+1,d_model))

        # Encoder separados

        self.spatial_encoder=TransformerEncoder(d_model=d_model,
                                                num_heads=num_heads,
                                                num_layers=num_layers,
                                                mlp_size=mlp_size
                                                )
        
        self.temporal_encoder=TransformerEncoder(d_model=d_model,
                                                num_heads=num_heads,
                                                num_layers=num_layers,
                                                mlp_size=mlp_size)
        
        self.mlp_head=nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            nn.Linear(in_features=d_model,out_features=num_classes)
        )

    
    def forward(self,x:torch.tensor):
        #print(x.shape)

        x=self.patch_embedding(x)

        bs,frames,_,_=x.shape

        #print(f"Shape after Patch Embedding: {x.shape}")

        spatial_token=self.spatial_cls_token.expand(bs,frames,-1,-1)

        #print(f"New dimension of the spatial class token : {spatial_token.shape}")

        x=torch.cat([spatial_token,x],dim=2)

        #print(f"Shape of X after the concatenation of the spatial token : {x.shape}")

        spatial_positional_embedding=self.spatial_pos_embedding.expand(bs,frames,-1,-1)

        #print(f"New dimension of the spatial positional embedding: {spatial_positional_embedding.shape}")

        x=x + spatial_positional_embedding

        x=x.flatten(0,1) # (bs*frames,n.d_model)

        #print(f"Dimension of the X after adding the positional embeddings: {x.shape}")

        x=self.spatial_encoder(x)

        _,_,d_emb=x.shape

        x=x.view(bs,frames,-1,d_emb)

        #print(f"Shape after the spatial encoder : {x.shape}")

        x=x[:,:,0]

        #print(f"Only the cls tokens: {x.shape}")

        temporal_token=self.temporal_cls_token.expand(bs,-1,-1)
        #temporal_token=self.temporal_cls_token

        #print(f"New shape of the temporal token: {temporal_token.shape}")

        x=torch.concat([temporal_token,x],dim=1)

        #print(f"Shape after adding the temporal token: {x.shape}")

        temporal_positional_embedding=self.temporal_cls_token.expand(bs,-1,-1)

        x=x+temporal_positional_embedding

        x=self.temporal_encoder(x)

        #print(f"Shape after the temporal encoder: {x.shape}")

        final_cls_token=x[:,0]

        #print(f"Shape of the final token: {final_cls_token.shape}")


        return self.mlp_head(final_cls_token)
    

def encode_frame_to_base64(frame:torch.Tensor):

    frame=transform_undo_resize(frame)

    frame=frame.detach().cpu()

    frame=frame.permute(1,2,0).numpy()

    frame=(frame * 255).astype(np.uint8)

    _, buffer = cv2.imencode(".jpg", frame)

    return base64.b64encode(buffer).decode("utf-8")

