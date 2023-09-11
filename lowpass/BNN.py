import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.stats import norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir="./run")

class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = Normal(0,prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape).to(DEVICE)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(DEVICE)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)

class MLP_BBB(nn.Module):
    def __init__(self, headsize, hidden_units,outsize, noise_tol=.1,  prior_var=1.):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.hidden = Linear_BBB(headsize, hidden_units, prior_var=prior_var)
        self.out = Linear_BBB(hidden_units, outsize, prior_var=prior_var)
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        #print('input size',x.size())
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        #print('output size',x.size())
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior  + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples, num):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        Outputs = torch.zeros(samples, target.reshape(-1).shape[0]).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_posts = torch.zeros(samples).to(DEVICE)
        log_likes = torch.zeros(samples).to(DEVICE)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            Outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(Outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = (log_post - log_prior)/num - log_like
        return loss


with open(r"../data/all.csv") as file:
        dataset = pd.read_csv(file)
        inputs = ["field.angular_velocity.x","field.angular_velocity.y","field.angular_velocity.z",
                  "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
        outputs = ["left_uwb_thread","right_uwb_thread"]

batch = 50
seclect_data = int(0.7*dataset.shape[0])
x_train = torch.tensor(dataset[inputs].iloc[:seclect_data,:].values).to(torch.float32).to(DEVICE)
y_train = torch.tensor(dataset[outputs].iloc[:seclect_data,:].values).to(torch.float32).to(DEVICE)

x_test = torch.tensor(dataset[inputs].iloc[seclect_data:,:].values).to(torch.float32).to(DEVICE)
y_test = torch.tensor(dataset[outputs].iloc[seclect_data:,:].values).to(torch.float32).to(DEVICE)

train_ds =TensorDataset(x_train,y_train)
train_dl =DataLoader(train_ds,batch_size = batch,shuffle = True)

valid_ds =TensorDataset(x_test,y_test)
valid_dl =DataLoader(valid_ds,batch_size = batch*2)

net = MLP_BBB(headsize=6,hidden_units=20,outsize=2, prior_var=1).to(DEVICE)
optimizer = optim.AdamW(net.parameters(), lr=1e-4)
epochs = 200*10000

drawloss = 0
drawvalid_loss = 0
best_model = 1e10

i = 0
for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
    for xb,yb in train_dl:
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = net.sample_elbo(xb, yb, 1,num = len(train_dl))
        loss.backward()
        optimizer.step()
    net.eval()
    for xb, yb in valid_dl:
        torch.no_grad()
        valid_loss = net.sample_elbo(xb, yb, 1,num = len(valid_dl))


    drawloss+=loss.item()
    drawvalid_loss+=valid_loss.item()
    if i %100==0:
        if valid_loss.item() < best_model:
            if os.path.exists('../models/net_best_model.pth'): os.remove('../models/net_best_model.pth')
            torch.save(net, '../models/net_best_model.pth')
            best_model = valid_loss.item()
        writer.add_scalar('training_loss',drawloss/100,epoch+1)
        writer.add_scalar('valid_loss', drawvalid_loss/100, epoch + 1)
        print('loss', drawloss/100, 'valid_loss', drawvalid_loss/100, 'epoch', epoch + 1,'best_model',best_model)
        drawloss,drawvalid_loss = 0, 0
    i+=1

print('Finished Training')
torch.save(net, './models/net'+datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")+'.pth')
print('best_model',best_model)