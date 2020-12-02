# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader
import torch
from utils import *
from datasets import *
from ConvS2S import *
from Attention import *
from Encoder import *
from trainer import *
from LiarLiar import *
# Your code goes here.
liar_dataset_train = dataset(prep_Data_from = 'train')
liar_dataset_val = dataset(prep_Data_from = 'val')
sentence_length, justification_length = liar_dataset_train.get_max_lenghts()
dataloader_train = DataLoader(dataset=liar_dataset_train, batch_size=25, num_workers=4)
dataloader_val = DataLoader(dataset=liar_dataset_val, batch_size=25, num_workers=4)
statement_encoder = Encoder(hidden_dim=512, conv_layers = 5)
justification_encoder = Encoder(hidden_dim=512, conv_layers = 5)
multiheadAttention = MultiHeadAttention(hid_dim = 512, n_heads = 32)
positionFeedForward = PositionFeedforward(512, 2048)
model = arePantsonFire(sentence_encoder=statement_encoder, explanation_encoder=justification_encoder, multihead_Attention=multiheadAttention, position_Feedforward=positionFeedForward, hidden_dim=512, max_length_sentence=sentence_length, max_length_justification=justification_length, input_dim=200, device='cuda:0')
#if num_epochs = 1 then it will run for 1 loop 
trainer(model, dataloader_train, dataloader_val, 1, path_to_save='/home/atharva', checkpoint_path='/home/atharva', checkpoint=100, train_batch=1, test_batch=1)            

# Do not change module_list , otherwise no marks will be awarded
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)
