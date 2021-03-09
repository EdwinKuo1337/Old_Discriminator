import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
import hdf5storage
from random import shuffle
import time
import os
import random
# from models.MDTGCN_3D_gru_attention_correct import ResNet, ResidualBlock
from models.Deasfn_discriminator_phase import *

from models import graph

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# batch_size = 32
num_epochs = 30
learning_rate = 0.01
gcnNum = 3
k = 2
num_frame = 5
frames_per_sample = 60
batch_sequence_size = 32 #batch_sequence_size = batch_size / num_frame

checkpoint_discriminator ='weights_AAAI/Deasfn_noise_discriminator_phase_12_05_1.pkl'
checkpoint_encoder ='weights_AAAI/Deasfn_noise_encoder_phase_12_05_1.pkl'


dilation = 3

def save_checkpoint(state, filename):
    torch.save(state, filename)
    # shutil.copyfile(filename, 'model_best.pth.tar')



def getSequenceMinibatch(file_names):
    sequence_num = len(file_names)
    csi_data = torch.zeros(sequence_num, num_frame, 30, 25, 3, 2)
    # csi_data = torch.zeros(sequence_num, num_frame, 30, 5, 3, 3)

    labels = torch.zeros(sequence_num).type(torch.LongTensor)
    # newOnes = ['08', '09', '10', '11']

    for i in range(sequence_num):
        for j in range(num_frame):
            data = hdf5storage.loadmat(file_names[i][0][j], variable_names={'csi_serial_phase'})
            csi_data[i, j, :, :, :, :] = torch.from_numpy(data['csi_serial_phase']).type(torch.FloatTensor).permute(1, 0, 2, 3)
        labels[i] = torch.from_numpy(np.array(file_names[i][1]))
        #print(labels[i])
    return csi_data, labels

def takeSecond(elem):
    return int(elem.split("/")[-1].split(".")[0])

matTrain = []

matTest = []

subjects = ['01', '02', '03', '04', '05', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
actions = ['walk', 'hand', 'run', 'jump']
for subject in subjects:
    for action in actions:
        matTrain.append(glob.glob('/data2/lab50_dataset/'+subject+'/'+action+'/train_phase/*.mat'))
        matTest.append(glob.glob('/data2/lab50_dataset/'+subject+'/'+action+'/test_phase/*.mat'))


#split training data
for i in range(len(matTrain)) :
    matTrain[i].sort(key = takeSecond)
    matTest[i].sort(key = takeSecond)


matsTrain_sequence = []
matsTest_sequence = []

for i in range(len(matTrain)) : 
    matsTrain_sequence.append([])
    for j in range(0, int(len(matTrain[i])*0.8), frames_per_sample): #per_sequence = every frames_per_sample/20 sec we choose 5 frame as per_sequence
        per_sequence = []
        for k in range(num_frame):
            per_sequence.append(matTrain[i][j+k*dilation])
        matsTrain_sequence[i].append(per_sequence)

for i in range(len(matTest)) :
    matsTest_sequence.append([])
    for j in range(0, int(len(matTest[i])*0.8), frames_per_sample):
        per_sequence = []
        for k in range(num_frame):
            per_sequence.append(matTest[i][j+k*dilation])
        matsTest_sequence[i].append(per_sequence)

trainingPairs = []


#trainingDatas = []
#train = 0-199
for i in range(len(matsTrain_sequence)): #i = 50(ppl) * 4(action)
    for k in range(int(0.8*len(matsTrain_sequence[i]))): #j*k = number of possible pairs per ppl per action.
        trainingPairs.append((matsTrain_sequence[i][k],i%4))

#for i in range(len(matsTest_sequence)): #i = 50(ppl) * 4(action)
    #for k in range(int(0.8*len(matsTest_sequence[i]))): #j*k = number of possible pairs per ppl per action.
        #trainingPairs.append((matsTest_sequence[i][k],(i+200)))
        
#test = 200-399
mats_sequence_num = len(trainingPairs)
batch_sequence_num = int(np.floor(mats_sequence_num/batch_sequence_size))
print('num_frame = ', num_frame)
print('mats_sequence_num = ', mats_sequence_num)
print('batch_sequence_num = ', batch_sequence_num)

# mats = []
# for i in range(len(mat)) :
#     for j in range(int(len(mat[i]) * 0.3)):
#         mats.append(mat[i][j])
#
#
# mats_num = len(mats)
# batch_num = int(np.floor(mats_num/batch_size))
# print(mats_num)


# wisppn = ResNet(ResidualBlock, [2, 2, 2, 2], adj, 3)
# wisppn = ResNet(ResidualBlock, [2, 2, 2, 2])
# wisppn = Deasfn()
encoder = encoder()
discriminator = classifier()

#encoder = filter(lambda p: p.requires_grad, encoder.parameters())
#discriminator = filter(lambda p: p.requires_grad, discriminator.parameters())

encoder.weights_init()
discriminator.weights_init()


for p in encoder.parameters():
    p.requires_grad = True
for p in discriminator.parameters():
    p.requires_grad = True

optimizer_encoder = torch.optim.Adam(encoder.cuda().parameters(), lr=learning_rate)
optimizer_discriminator = torch.optim.Adam(discriminator.cuda().parameters(), lr=learning_rate)



if checkpoint_encoder :
    if os.path.isfile(checkpoint_encoder):
        print("=> loading encoder checkpoint '{}'".format(checkpoint_encoder))
        checkpoint = torch.load(checkpoint_encoder)
        start_epoch_e = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['state_dict'])
        optimizer_encoder.load_state_dict(checkpoint['optimizer'])
        print("=> loaded encoder checkpoint '{}' (epoch {})"
              .format(checkpoint_encoder, checkpoint['epoch']))
    else:
        start_epoch_e = 0
        print("=> no encoder checkpoint found at '{}'".format(checkpoint_encoder))

if checkpoint_discriminator:
    if os.path.isfile(checkpoint_discriminator):
        print("=> loading discriminator checkpoint '{}'".format(checkpoint_discriminator))
        checkpoint = torch.load(checkpoint_discriminator)
        start_epoch_d = checkpoint['epoch']
        discriminator.load_state_dict(checkpoint['state_dict'])
        optimizer_discriminator.load_state_dict(checkpoint['optimizer'])
        print("=> loaded discriminator checkpoint '{}' (epoch {})"
              .format(checkpoint_discriminator, checkpoint['epoch']))
    else:
        start_epoch_d = 0
        print("=> no discriminator checkpoint found at '{}'".format(checkpoint_discriminator))

CrossEntropy = nn.CrossEntropyLoss()

encoder = encoder.cuda()
discriminator = discriminator.cuda()

scheduler_e = torch.optim.lr_scheduler.MultiStepLR(optimizer_encoder, milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], gamma=0.5)
scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_discriminator, milestones=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], gamma=0.5)

encoder.train()
discriminator.train()

start_epoch = min(start_epoch_e, start_epoch_d)

for epoch_index in range(start_epoch,num_epochs):

    print('epoch_index=', epoch_index)

    start = time.time()

    # shuffling dataset
    shuffle(trainingPairs)
    loss_per_epoch = 0
    #per10 = 0
    # in each minibatch
    for batch_index in range(batch_sequence_num):
        if batch_index < batch_sequence_num:
            file_names = trainingPairs[batch_index*batch_sequence_size:(batch_index+1)*batch_sequence_size]
        else:
            file_names = trainingPairs[batch_sequence_num*batch_sequence_size:]

        # csi_data, heatmaps = getMinibatch(file_names)
        csi_data, labels = getSequenceMinibatch(file_names)
        #print('csi_data = ', csi_data.size())

        csi_data = Variable(csi_data.cuda())
        #print('csi_data = ', csi_data.size())
        

        labels = Variable(labels.cuda())
        
		#print('csi_data = ', csi_data.shape)
		
        b, t, c, f, h, w = csi_data.size()
        csi_data = csi_data.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)
        #print(csi_data.size())
        out = encoder(csi_data)
        #print(out.size())
        prediction = discriminator(out,b,t)
        #print(prediction)
        #print(labels)
        loss = CrossEntropy(prediction, labels)
        #print(loss)
        loss_per_epoch += loss.item()
        #print(loss.item())
        """
        if per10 < 9 :
            per10 +=1
        else :
            per10 = 0


        if per10 == 9 :
            print("total " + str(loss.item()))
		"""
        optimizer_encoder.zero_grad()
        optimizer_discriminator.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_discriminator.step()

	
    print('epoch = ', epoch_index,' loss_per_epoch = ',loss_per_epoch/batch_sequence_num)
    scheduler_e.step()
    scheduler_d.step()
    endl = time.time()
    print('Costing time:', (endl-start)/60)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    save_checkpoint({
        'epoch': epoch_index + 1,
        # 'arch': args.arch,
        'state_dict': encoder.state_dict(),
        # 'best_prec1': best_prec1,
        'optimizer': optimizer_encoder.state_dict(),
    },checkpoint_encoder)
    save_checkpoint({
        'epoch': epoch_index + 1,
        # 'arch': args.arch,
        'state_dict': discriminator.state_dict(),
        # 'best_prec1': best_prec1,
        'optimizer': optimizer_discriminator.state_dict(),
    },checkpoint_discriminator)
