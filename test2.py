import pickle
import numpy as np
import getInput

a=1
#while(a<=98):
#ds = pickle.load(open("/home/dhanya/Documents/Machine Learning/Neural Networks/RNN/MFCC/fours/f"+str(a)+"_mfcc.p","rb"))
# ds is np.ndarray type.


#print(inputs)
# ds is a vector of length 1098, each element is a vector of length 13. One RNN unit takes one 13 element vector as input.

input_size = 13 
vocab_size = input_size
output_size = 4

# hyperparameters
hidden_size = 16 # size of hidden layer of neurons
###
seq_length = 8 # number of steps to unroll the RNN for

## reduce the learning rate and try.
learning_rate = 0.3

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((output_size, 1)) # output bias
#print("by",by)
#print(len(getInput.trainf))

'''def gInput(getInput.trainf):
  for eachEight in getInput.trainf:
    for t in range(len(eachEight)): # 8
      for i in range(len(eachEight[t])):
        temp = np.zeros(shape = (13,1))
        temp[i] = eachEight[t][i]'''
      

def test(eachEight, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  
  for t in range(len(eachEight)):
      temp = np.zeros(shape = (13,1))   # 8
      for i in range(len(eachEight[t])):
        
        temp[i] = eachEight[t][i]
      #print("raw",inputs[t])
      xs[t] = temp # one vector, t = 0-7
      #print("input one",xs[t],len(xs[t]))
      hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
      #print("hs",hs[t],len(hs[t]))
      #print(Why,"Why",len(Why))
      # not so far so good :D
      temp = np.dot(Why, hs[t])
      #print(temp,"temp",len(temp))
      ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
      
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      #print("ps",ps[t],len(ps[t]))
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  return loss
      


def lossFun(eachEight, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  
  for t in range(len(eachEight)):
      temp = np.zeros(shape = (13,1))   # 8
      for i in range(len(eachEight[t])):
        
        temp[i] = eachEight[t][i]
      #print("raw",inputs[t])
      xs[t] = temp # one vector, t = 0-7
      #print("input one",xs[t],len(xs[t]))
      hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
      #print("hs",hs[t],len(hs[t]))
      #print(Why,"Why",len(Why))
      # not so far so good :D
      temp = np.dot(Why, hs[t])
      #print(temp,"temp",len(temp))
      ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
      
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      #print("ps",ps[t],len(ps[t]))
      loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(seq_length)):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y 
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[seq_length-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

lossf = []
losss = []
losso = []

while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  #inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  #targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  #print("input",len(inputs))
  #print("targets",targets)
  
    
  # sample from the model now and then
  '''if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )'''
  
  # forward seq_length characters through the net and fetch gradient

# fours = 0
  for eachEight in getInput.trainf1:
      targets = [3,3,3,3,3,3,3,0]
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(eachEight, targets, hprev)
      print("loss fours train",loss)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 

#outs = 1
  for eachEight in getInput.trainf2:
      targets = [3,3,3,3,3,3,3,1]
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(eachEight, targets, hprev)
      print("loss outs train",loss)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 

#sixes = 2
  for eachEight in getInput.trainf3:
      targets = [3,3,3,3,3,3,3,2]
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(eachEight, targets, hprev)
      print("loss sixes train",loss)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 

###########################################   testing!!!



  for eachEight in getInput.testf1:
      targets = [3,3,3,3,3,3,3,0]
      loss = test(eachEight, targets, hprev)
      lossf.append(loss)
      print("loss fours test",loss)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 

# outs = 1
  for eachEight in getInput.testf2:
      targets = [3,3,3,3,3,3,3,1]
      loss = test(eachEight, targets, hprev)
      losss.append(loss)
      print("loss outs test",loss)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 

#sixes = 2
  for eachEight in getInput.testf3:
      targets = [3,3,3,3,3,3,3,2]
      loss = test(eachEight, targets, hprev)
      losso.append(loss)
      print("loss sixes test",loss)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 

  

