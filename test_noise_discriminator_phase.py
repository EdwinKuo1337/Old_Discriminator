import numpy as np
import torch
from torch.autograd import Variable
import hdf5storage
import os
from models.Deasfn_discriminator_phase import *
import glob
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
num_frame = 5
sequence_num = 1

def takeSecond(elem):
    return int(elem.split("/")[-1].split(".")[0])

encoder = encoder()
discriminator = classifier()

checkpoint_discriminator =torch.load('weights_AAAI/Deasfn_noise_discriminator_phase_12_05_1.pkl')
checkpoint_encoder =torch.load('weights_AAAI/Deasfn_noise_encoder_phase_12_05_1.pkl')

encoder.load_state_dict(checkpoint_encoder['state_dict'])
encoder = encoder.cuda().eval()
discriminator.load_state_dict(checkpoint_discriminator['state_dict'])
discriminator = discriminator.cuda().eval()

dilation = 3
"""
action = 'run'
subjects = ['14']
"""
range_list = [range(600, 700)]

matTrain = []
matTest = []

subjects = ['01', '02', '03', '04', '05', '07', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
actions = ['walk', 'hand', 'run', 'jump']
for subject in subjects:
    for action in actions:
        matTrain.append(glob.glob('/data2/lab50_dataset/'+subject+'/'+action+'/train_phase/*.mat'))
        #matTest.append(glob.glob('/data2/lab50_dataset/'+subject+'/'+action+'/test_phase/*.mat'))


for i in range(len(matTrain)) :
    matTrain[i].sort(key = takeSecond)
    #matTest[i].sort(key = takeSecond)


matsTrain_sequence = []
#matsTest_sequence = []

for i in range(len(matTrain)) :
    matsTrain_sequence.append([])
    for j in range(0, int(len(matTrain[i])*0.8), 60):
        per_sequence = []
        for k in range(num_frame):
            per_sequence.append(matTrain[i][j+k*dilation])
        matsTrain_sequence[i].append(per_sequence)

'''
for i in range(len(matTest)) :
    matsTest_sequence.append([])
    for j in range(int(len(matTest[i])*0.7), int(len(matTest[i])*0.9), 20):
        per_sequence = []
        for k in range(num_frame):
            per_sequence.append(matTest[i][j+k*dilation])
        matsTest_sequence[i].append(per_sequence)
'''
testingPairs = []


for i in range(len(matsTrain_sequence)):
    for j in range(int(len(matsTrain_sequence[i]))):
        label = i%4
        pair = random.choice(matsTrain_sequence[i])
        testingPairs.append((pair, label))

     
cnt = 0
#print('testingPains = ', len(testingPairs))
for i in range(len(testingPairs)):
    #print(testingPairs[i][2])
    #print('testingPains = ', len(testingPairs))
    csi_data = torch.zeros(1, num_frame, 30, 25, 3, 2)
    for j in range(num_frame):
        data = hdf5storage.loadmat(testingPairs[i][0][j], variable_names={'csi_serial_phase'})
        csi_data[0, j, :, :, :, :] = torch.from_numpy(data['csi_serial_phase']).type(torch.FloatTensor).permute(1, 0, 2, 3)
    csi_data = Variable(csi_data.cuda())
    
    b, t, c, f, h, w = csi_data.size()
    csi_data = csi_data.permute(1, 0, 2, 3, 4, 5).reshape(-1, c, f, h, w)

    out = encoder(csi_data)
    prediction = discriminator(out,b,t)
    
    if prediction.cpu().detach().numpy().argsort()[-1][-1] == testingPairs[i][1] :
        cnt += 1
        #print(cnt)
    #exit()
print(cnt, ' / ', len(testingPairs))
print('correctness = ', cnt/len(testingPairs)*100, '%')      
        
