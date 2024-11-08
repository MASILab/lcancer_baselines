import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import lungbl.dlstm.crnn as crnn
import lungbl.dlstm.rnn as myrnn

RNNS = ['LSTM', 'GRU']



class CNNNet(nn.Module):
    def __init__(self, step, drop):
        super(CNNNet, self).__init__()
        #self.embedding = embedding

        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.cnn2d = nn.Conv2d( 5, 5, 2, stride = 2, padding = 0)
        self.cnn1d = nn.Conv1d(10, 5, 5, stride = 4)
        self.fc2 = nn.Linear(15,1) 
        self.drop = drop
        

        size = 0
        for p in self.parameters():
          size += p.nelement()
        print('Total param size: {}'.format(size))
        
    def forward(self, x):
        step, bsize, num_nodule, dim = x.shape
        x = x.permute([1,2,0, 3])
        x = torch.reshape(x, (bsize, num_nodule * step, dim))
        
        x = self.cnn1d(x)
        if self.drop:
            x = F.dropout(x, training=self.training)
        #print (x.shape)
        #x = torch.squeeze(x,2)
        
        
        #x = self.cnn1d2(x)
        #x = nn.MaxPool1d(3)(x)
        #print ('------x.shape---', x.shape)
        
        
        
        #x = torch.sigmoid(self.fc2(x))
        #out2 = x.view(x.size(0), -1)
 
        out2 = torch.sigmoid(self.fc2(x))
        #print ('out2.shape', out2.shape)
        out2 = torch.squeeze(out2, -1)
        #casePred = 
        base_prob = torch.sigmoid(self.baseline)
        casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))
        return casePred
    

    
class RNNCellNet(nn.Module):
  def __init__(self):
    super(RNNCellNet, self).__init__()
    #self.embedding = embedding

    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    self.fc2 = nn.Linear(64,1)
    self.lstmcell = nn.LSTMCell(64, 64)
    #self.decoder = nn.Linear(hidden_dim, 1)
    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))

    
  def forward(self, input):                        # input should be [bsize, 5, dim]
    #print (input.data.cpu().numpy().shape)
    step, bsize, num_nodule, dim = input.shape
    input = torch.reshape(input, (step, bsize * num_nodule, dim))
    for i in range(step):
        if i == 0:
            hx, cx = self.lstmcell(input[i])
        else:
            hx, cx = self.lstmcell(input[i], (hx, cx))
    
     
    out2 = torch.sigmoid(self.fc2(hx))

    out2 = out2.reshape(bsize, num_nodule)
    #print (out2.shape)
    base_prob = torch.sigmoid(self.baseline)
    casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))
    return casePred

class DisRNNCellNet(nn.Module):
  def __init__(self, mode, drop = False):
    super(DisRNNCellNet, self).__init__()
    #self.embedding = embedding

    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    self.fc2 = nn.Linear(64,1)
    self.dislstmcell = myrnn.dLSTMCell(64, 64, mode = mode)
    #self.decoder = nn.Linear(hidden_dim, 1)
    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))
    self.drop = drop


  def forward(self, input, time_dis):                       
    #print (input.data.cpu().numpy().shape)
    step, bsize, num_nodule, dim = input.shape
    #print ('time.shape', time_dis.shape)
    time_dis = time_dis.unsqueeze(2).expand(bsize, step, num_nodule)
    #print ('time.shape', time_dis.shape)
    input = torch.reshape(input, (step, bsize * num_nodule, dim))
    time_dis = time_dis.permute(1, 0, 2)
    time_dis = torch.reshape(time_dis, (step, bsize * num_nodule))
    #print ('time.shape', time_dis.shape)
    for i in range(step):
        if i == 0:
            hx, cx = self.dislstmcell(input[i], time_dis[i])
        else:
            hx, cx = self.dislstmcell(input[i], time_dis[i - 1], (hx, cx))
    if self.drop:
        hx = F.dropout(hx, p = 0.8, training=self.training)
     
    out2 = torch.sigmoid(self.fc2(hx))

    out2 = out2.reshape(bsize, num_nodule)
    #print (out2.shape)
    base_prob = torch.sigmoid(self.baseline)
    casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))
    return casePred


class CRNNClassifier(nn.Module):
  def __init__(self, time, drop):
    super(CRNNClassifier, self).__init__()
    #self.embedding = embedding
    self.lstmcell = crnn.Conv1dLSTMCell(in_channels = 5, out_channels = 5, kernel_size = 5)
    self.grucell = crnn.Conv1dGRUCell(in_channels=5, out_channels=5, kernel_size=5)

    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    self.fc2 = nn.Linear(64,1)
    self.drop = drop
    self.time = time
    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))

  def forward(self, x):                        # input should be [bsize, 5, dim]
    #print (input.data.cpu().numpy().shape)
    for i in range(self.time):
            if i == 0:
                hx, cx = self.lstmcell(x[i])
            else:
                hx, cx = self.lstmcell(x[i], (hx, cx))
#     print ("use gru")
#     for i in range(self.time):
#         if i == 0:
#             hx = self.grucell(x[i])
#         else:
#             hx = self.grucell(x[i], hx)
    if self.drop:
        x = F.dropout(hx, training=self.training)
    else:
        x = hx
    
    x = torch.sigmoid(self.fc2(x))
    
    out2 = torch.squeeze(x, -1)
    #print ('out2.shape', out2.shape)
    base_prob = torch.sigmoid(self.baseline)
    casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))

    return casePred



class MCRNNClassifier(nn.Module):
  def __init__(self, time, drop):
    super(MCRNNClassifier, self).__init__()
    #self.embedding = embedding
    #self.lstmcell = crnn.Conv1dLSTMCell(in_channels = 5, out_channels = 5, kernel_size = 5)
    self.grucell = crnn.Conv1dGRUCell(in_channels = 5, out_channels = 5, kernel_size = 5)

    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    self.fc1 = nn.Linear(128, 64)
    self.fc2 = nn.Linear(64,1)
    self.drop = drop
    self.time = time
    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))

  def forward(self, x0):                        # input should be [bsize, 5, dim]
    #print (input.data.cpu().numpy().shape)
    x1 = x0[[1,0]]
    assert self.time == 2
    inp = []
    # for x in [x0, x1]:
    #     for i in range(self.time):
    #         if i == 0:
    #             hx, cx = self.lstmcell(x[i])
    #         else:
    #             hx, cx = self.lstmcell(x[i], (hx, cx))
    #     inp.append(hx)

    for x in [x0, x1]:
        print ('----- use gru ------------')
        for i in range(self.time):
            if i == 0:
                hx = self.grucell(x[i])
            else:
                hx = self.grucell(x[i], hx)
        inp.append(hx)

    x = torch.cat(inp, 2)
    if self.drop:
        x = F.dropout(x, training=self.training)
    else:
        x = x
    #print ('---x.shape---', x.shape)
    x = torch.sigmoid(self.fc1(x))
    out2 = torch.sigmoid(self.fc2(x))
    out2 = torch.squeeze(out2, -1)
    #print ('out2.shape', out2.shape)
    base_prob = torch.sigmoid(self.baseline)
    casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))

    return casePred


class DisCRNNClassifier(nn.Module):
  def __init__(self, time, drop, mode, dim=64):
    super(DisCRNNClassifier, self).__init__()
    #self.embedding = embedding
    self.dislstmcell = crnn.LSTMdistCell(mode, in_channels = 5, out_channels = 5, kernel_size = 5, convndim = 1)
    self.drop = drop
    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    self.fc2 = nn.Linear(dim,1)
    self.time = time
    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))

  def forward(self, x, time_dist):                        
    # x.shape = (bsize, sequence length, 5, dim)
    x = x.permute([1, 0, 2, 3]) # time dim first
    
    for i in range(self.time):
        #time_dist[:, i] = torch.Tensor([3.]).cuda() * time_dist[:, i]
        #print (time_dist[:, i])
        if i == 0:
            hx, cx = self.dislstmcell(x[i], [time_dist[:,0], time_dist[:,0]])
            #hx, cx = self.dislstmcell(x[i], time_dist[:,0])
        else:
            hx, cx = self.dislstmcell(x[i], [time_dist[:,i-1], time_dist[:,i]], (hx, cx)) # if use infor, time_dist[:,i], use dis_exp, time_dist[:,i-1]
    
    if self.drop:
        x = F.dropout(hx, training=self.training)
    else:
        x = hx
    out2 = torch.sigmoid(self.fc2(x))
    out2 = torch.squeeze(out2, -1)
    #print ('out2.shape', out2.shape)
    base_prob = torch.sigmoid(self.baseline)
    casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))

    return casePred, out2



class CTLSTM(nn.Module): # not test yet
  def __init__(self, time, drop):
    super(CTLSTM, self).__init__()
    #self.embedding = embedding
    self.dislstmcell = crnn.TLSTMCell('TLSTMv2', in_channels = 5, out_channels = 5, kernel_size = 5, convndim = 1)
    self.drop = drop
    self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
    self.fc2 = nn.Linear(64,1)
    self.time = time
    size = 0
    for p in self.parameters():
      size += p.nelement()
    print('Total param size: {}'.format(size))

  def forward(self, x, time_dist):                        # input should be [bsize, 5, dim]
    #print (input.data.cpu().numpy().shape)
    
    for i in range(self.time):
        #time_dist[:, i] = torch.Tensor([3.]).cuda() * time_dist[:, i]
        #print (time_dist[:, i])
        if i == 0:
            hx, cx = self.dislstmcell(x[i], [time_dist[:,0], time_dist[:,0]])
            #hx, cx = self.dislstmcell(x[i], time_dist[:,0])
        else:
            hx, cx = self.dislstmcell(x[i], [time_dist[:,i-1], time_dist[:,i]], (hx, cx)) # if use infor, time_dist[:,i], use dis_exp, time_dist[:,i-1]
    
    if self.drop:
        x = F.dropout(hx, training=self.training)
    else:
        x = hx
    out2 = torch.sigmoid(self.fc2(x))
    out2 = torch.squeeze(out2, -1)
    #print ('out2.shape', out2.shape)
    base_prob = torch.sigmoid(self.baseline)
    casePred = 1-torch.prod(1-out2,dim=1)*(1-base_prob.expand(out2.size()[0]))

    return casePred

class TimeLSTM(nn.Module):
      def __init__(self, time, drop):
          super(TimeLSTM, self).__init__()
          # self.embedding = embedding
          self.dislstmcell = crnn.TimeLSTMCell('TimeLSTM', in_channels=5, out_channels=5, kernel_size=5, convndim=1)
          self.drop = drop
          self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
          self.fc2 = nn.Linear(64, 1)
          self.time = time
          size = 0
          for p in self.parameters():
              size += p.nelement()
          print('Total param size: {}'.format(size))

      def forward(self, x, time_dist):  # input should be [bsize, 5, dim]
          # print (input.data.cpu().numpy().shape)

          for i in range(self.time):
              # time_dist[:, i] = torch.Tensor([3.]).cuda() * time_dist[:, i]
              # print (time_dist[:, i])
              if i == 0:
                  hx, cx = self.dislstmcell(x[i], [time_dist[:, 0], time_dist[:, 0]])
                  # hx, cx = self.dislstmcell(x[i], time_dist[:,0])
              else:
                  hx, cx = self.dislstmcell(x[i], [time_dist[:, i - 1], time_dist[:, i]],
                                            (hx, cx))  # if use infor, time_dist[:,i], use dis_exp, time_dist[:,i-1]

          if self.drop:
              x = F.dropout(hx, training=self.training)
          else:
              x = hx
          out2 = torch.sigmoid(self.fc2(x))
          out2 = torch.squeeze(out2, -1)
          # print ('out2.shape', out2.shape)
          base_prob = torch.sigmoid(self.baseline)
          casePred = 1 - torch.prod(1 - out2, dim=1) * (1 - base_prob.expand(out2.size()[0]))

          return casePred
