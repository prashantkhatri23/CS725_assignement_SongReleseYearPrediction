import sys
import os
import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

class Net(object):

  def __init__(self, num_layers, num_units):
    '''
    Initialize the neural network.
    Create weights and biases.

    Here, we have provided an example structure for the weights and biases.
    It is a list of weight and bias matrices, in which, the
    dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
    weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, 1)]
    biases: [(num_units, 1), (num_units, 1), (num_units, 1)]

    Please note that this is just an example.
    You are free to modify or entirely ignore this initialization as per your need.
    Also you can add more state-tracking variables that might be useful to compute
    the gradients efficiently.


    Parameters
    ----------
    num_layers : Number of HIDDEN layers
    num_units : Number of units in each Hidden layer.

    used He initlization
    '''
    self.num_layers = num_layers #number of hidden layers
    self.num_units = num_units #number of neurons in each hidden layer
    self.biases = []
    self.weights = []
    for i in range(self.num_layers):
      if i==0:
        # Input layer
        self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))

      else:
        # Hidden layer
        self.weights.append(np.random.uniform(-1, 1, size=(num_units, self.num_units)))

      self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

    # Output layer
    self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
    self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
    self.scaler_X = (0,1)
    self.scaler_y = (0,1)


  def __call__(self, X):
    '''
    Forward propagate the input X through the network,
    and return the output.

    Note that for a classification task, the output layer should
    be a softmax layer. So perform the computations accordingly

    Parameters
    ----------
      X : Input to the network, numpy array of shape m x d
    Returns
    ----------
      y_hat : Output of the network, numpy array of shape m x 1
    '''
    X = transform_standard_scaler_fun(X,self.scaler_X)
    Lrelu_slope = 0
    a = X
    L = self.num_layers + 1 #number of weight matrixes, total layers-1
    A = [a] #stores the activaiton values for each node
    del_A_h = [np.ones_like(a)] #stores values of the derivative of the activation for each node

    for i, (w,b) in enumerate(zip(self.weights,self.biases)):
      l = i+1 #considering input layer as 0th layer
      h = np.dot(a,w) + b.T
      if l <= L-1:
        a = np.maximum(h,Lrelu_slope*h) #relu(h)
        A += [a]
        del_A_h += [(h>=0)*1+(h<0)*Lrelu_slope] #del_relu(h) i.e derivative of relu(x) at x=h
      else:
        a = h
        A += [a]
        del_A_h += [np.ones_like(a)]

    y_hat = A[L] #prediction
    y_hat = inverse_standard_scaler_fun(y_hat,self.scaler_y)
    return y_hat,A,del_A_h

  def backward(self, X, y, lamda):
    '''
    Compute and return gradients loss with respect to weights and biases.
    (dL/dW and dL/db)

    Parameters
    ----------
    X : Input to the network, numpy array of shape m x d
    y : desired output of the network, numpy array of shape m x 1
    lamda : Regularization parameter.

    Returns
    ----------
    del_W : derivative of loss w.r.t. all weight values (a list of matrices).
    del_b : derivative of loss w.r.t. all bias values (a list of vectors).

    Hint: You need to do a forward pass before performing backward pass.
		'''
    X = np.array(X)
    y = np.array(y)
    M = X.shape[0] # number of the training instnaces
    y_pre,A,A_del = self(X)
    y_pre = transform_standard_scaler_fun(y_pre,self.scaler_y)

    y = np.reshape(y,(len(y),1)) #converitng to row matrix in case not
    y = transform_standard_scaler_fun(y,self.scaler_y)

    del_W = []
    del_b = []
    L = self.num_layers + 1  #number of weight matrixes
    S = [(y_pre-y)] # stores the sensitivity values
    for l in np.arange(2,L+1)[::-1]: # L to 2
      sensitivity = S[0].dot(self.weights[l-1].T) #indexing of the weights starts form 1
      sensitivity = sensitivity*A_del[l-1] #indexing of the activaitons starts form 0
      S = [sensitivity] + S #appends form front
    # S is of length L => indexed from 0 to L-1
    # A is of length L+1 => indexed from 0 to L
    for l in range(L): # 0 to L-1
      del_W += [A[l].T.dot(S[l])/M +lamda*self.weights[l]/M]
      d_b = np.mean(S[l],axis=0) + lamda*self.biases[l].T/M
      del_b += [d_b.T]
    return del_W,del_b

class Optimizer(object):

  def __init__(self, learning_rate):
    '''
    Create a Gradient Descent based optimizer with given
    learning rate.

    Other parameters can also be passed to create different types of
    optimizers.

    Hint: You can use the class members to track various states of the
    optimizer.
    '''
    self.learning_rate = learning_rate

  def step(self, weights, biases, delta_weights, delta_biases):
    '''
    Parameters
    ----------
      weights: Current weights of the network.
      biases: Current biases of the network.
      delta_weights: Gradients of weights with respect to loss.
      delta_biases: Gradients of biases with respect to loss.
    '''
    weights_updated = []
    biases_updated = []
    # print("biases,delta_biases = ",biases, delta_biases[0])
    for i in range(len(weights)):
      weights_updated  += [weights[i]-self.learning_rate*delta_weights[i]]
      biases_updated += [biases[i]-self.learning_rate*delta_biases[i]]
    # print("after update, biases = ",biases_updated.shape)
    return weights_updated,biases_updated

def loss_mse(y, y_hat):
  '''
  Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

  Parameters
  ----------
    y : targets, numpy array of shape m x 1
    y_hat : predictions, numpy array of shape m x 1

  Returns
  ----------
    MSE loss between y and y_hat.
  '''
  MSE = np.mean((y-y_hat)**2)

  return MSE

def loss_regularization(weights, biases):
  '''
  Compute l2 regularization loss.

  Parameters
  ----------
    weights and biases of the network.

  Returns
  ----------
    l2 regularization loss
  '''
  reg_loss = 0
  for w in weights:
    reg_loss += np.sum(w**2)
  return reg_loss

def loss_fn(y, y_hat, weights, biases, lamda):
  '''
  Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

  Parameters
  ----------
    y : targets, numpy array of shape m x 1
    y_hat : predictions, numpy array of shape m x 1
    weights and biases of the network
    lamda: Regularization parameter

  Returns
  ----------
    l2 regularization loss
  '''
  loss = (loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases) *(1/y.shape[0]))/2
  return loss

def rmse(y, y_hat):
  '''
  Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

  Parameters
  ----------
    y : targets, numpy array of shape m x 1
    y_hat : predictions, numpy array of shape m x 1

  Returns
  ----------
    RMSE between y and y_hat.
  '''
  RMSE = loss_mse(y, y_hat)**0.5
  return RMSE

def cross_entropy_loss(y, y_hat):
  '''
  Compute cross entropy loss

  Parameters
  ----------
    y : targets, numpy array of shape m x 1
    y_hat : predictions, numpy array of shape m x 1

  Returns
  ----------
    cross entropy loss
  '''
  CEL = -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
  return CEL

def fit_standard_scaler_fun(X):
  """
  for input matrix it returns a tuple of size two with first element being a 1D array containing mean of columns of the  mateix
  and second element as the varience of the columns of the input matrix.
  """
  return np.mean(X,axis=0),np.std(X,axis=0)

def transform_standard_scaler_fun(X,scaler):
  """
  for input matrix and a tuple with first element being a 1D array containing mean of columns of the some fitted matrix
  and second element as the varience of the columns of same matrix. The output is standard scaled matrix X
  """
  return (X-scaler[0])/scaler[1]

def inverse_standard_scaler_fun(X,scaler):
  """
  invert the standard scaled transform
  """
  return X*scaler[1]+scaler[0]

def train(net, optimizer, lamda, batch_size, max_epochs,
          train_input, train_target,dev_input, dev_target,
          UP_c=6,per_x_epoch = 2):
  '''
  In this function, you will perform following steps:
    1. Run gradient descent algorithm for `max_epochs` epochs.
    2. For each bach of the training data
      1.1 Compute gradients
      1.2 Update weights and biases using step() of optimizer.
    3. Compute RMSE on dev data after running `max_epochs` epochs.

  UP_c :  number of consecutive decrease after 'per_x_epoch' epochs # ref: Early Stopping — But When?
  per_x_epoch : length sequence of epochs after which to check the increment in dev loss


  Here we have added the code to loop over batches and perform backward pass
  for each batch in the loop.
  For this code also, you are free to heavily modify it.
  '''
  net.scaler_X = fit_standard_scaler_fun(train_input)
  net.scaler_y = fit_standard_scaler_fun(train_target)

  train_data = np.append(train_input,train_target,axis=1)

  m = train_input.shape[0]
  epoch_train_RegMSE_lst = []
  epoch_dev_RMSE_lst = []

  #stores the best parameters
  best_weights = net.weights
  best_biases = net.biases
  min_dev_loss = -1
  UP_count = UP_c #to track the decrease in the dev st error


  for e in range(max_epochs):
    epoch_loss = 0.
    np.random.shuffle(train_data)
    train_input = train_data[:,:-1]
    train_target = np.reshape(train_data[:,-1],(m,1))
    for i in range(0, m, batch_size):
      batch_input = train_input[i:i+batch_size]
      batch_target = train_target[i:i+batch_size]
      pred = net(batch_input)[0]
      # Compute gradients of loss w.r.t. weights and biases
      dW, db = net.backward(batch_input, batch_target, lamda)
      # Get updated weights based on current weights and gradients
      weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)
      # Update model's weights and biases
      net.weights = weights_updated
      net.biases = biases_updated
      # Compute loss for the batch
      batch_target_s = transform_standard_scaler_fun(batch_target,net.scaler_y)
      pred_s = transform_standard_scaler_fun(pred,net.scaler_y)
      batch_loss = loss_fn(batch_target_s, pred_s, net.weights, net.biases, lamda)
      epoch_loss += batch_loss
      if e%20==0 and i%12000<=0*batch_size:
        print("\ne, i, rmse_batch, batch_loss =",e, i, rmse(batch_target, pred), batch_loss)
    epoch_train_pred = net(train_input)[0]
    epoch_train_RegMSE_lst += [loss_fn(train_target, epoch_train_pred, net.weights, net.biases, lamda)]#[epoch_loss]
    epoch_dev_pred = net(dev_input)[0]
    epoch_dev_RMSE_lst += [rmse(dev_target, epoch_dev_pred)]
    if e%20==0:
      print("\ne, epoch_loss,epoch_dev_loss =",e, epoch_loss,epoch_dev_RMSE_lst[-1])
    if min_dev_loss==-1 or min_dev_loss > epoch_dev_RMSE_lst[-1]:
      min_dev_loss =  epoch_dev_RMSE_lst[-1]
      best_weights = net.weights
      best_biases = net.biases
    if e%per_x_epoch==0 and e>=per_x_epoch :
      if epoch_dev_RMSE_lst[-1] > epoch_dev_RMSE_lst[-1-per_x_epoch]:
        UP_count -= 1
        print("Print UP count = ",UP_c-UP_count)
        if UP_count==0:
          print("BREAK at epoch",e)
          break
      else:
        UP_count = UP_c
  net.biases = best_biases
  net.weights = best_weights
  dev_pred = net(dev_input)[0]
  dev_rmse = rmse(dev_target, dev_pred)

  train_pred = net(train_input)[0]
  train_regmse = loss_fn(train_target, epoch_train_pred, net.weights, net.biases, lamda)

  print('RMSE on dev data: {:.5f}'.format(dev_rmse))
  print('RegMSE on train data: {:.5f}'.format(train_regmse))
  print("min_dev_loss = ",min_dev_loss,net.num_layers, net.num_units)
  return net,min_dev_loss,epoch_train_RegMSE_lst,epoch_dev_RMSE_lst


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	y_hat = net(inputs)[0]
	predictions = y_hat
	prediction_file = "22M1079.csv"
	df = pd.DataFrame(predictions,index=range(1,predictions.shape[0]+1),columns=['Predictions'])
	df.to_csv(prediction_file,index_label="Id")
	return predictions

def read_data():
  '''
  Read the train, dev, and test datasets
  '''
  dev_file = "/22m1079/regression/data/dev.csv"
  dev_data = pd.read_csv(dev_file)
  dev_target = dev_data['1'].copy()
  dev_input = dev_data.drop('1',axis=1)
  dev_target = np.reshape(np.array(dev_target),(len(dev_target),1))
  dev_input = np.array(dev_input)

  train_file = "/22m1079/regression/data/train.csv"
  train_data = pd.read_csv(train_file)
  train_data = train_data.sample(frac=1)
  train_target = train_data['1'].copy()
  train_input = train_data.drop('1',axis=1)
  train_input = np.array(train_input)
  train_target = np.reshape(np.array(train_target),(len(train_target),1))

  test_file = "/22m1079/regression/data/test.csv"
  test_data = pd.read_csv(test_file)
  test_input = np.array(test_data)


  return train_input, train_target, dev_input, dev_target, test_input


def main():

  # Hyper-parameters
  max_epochs = 150
  batch_size = 64
  learning_rate = 0.01
  num_layers = 4
  num_units = 7
  lamda = 0.1 # Regularization Parameter
  Up_c = 6
  per_x_epoch = 2

  train_input, train_target, dev_input, dev_target, test_input = read_data()
  net = Net(num_layers, num_units)
  optimizer = Optimizer(learning_rate)
  train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target,Up_c,per_x_epoch
  )
  get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
