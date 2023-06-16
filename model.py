import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
from utils import GetCorrectPredCount
from matplotlib import pyplot as plt


class Net1(nn.Module):
    #This defines the structure of the NN.
    
    def __init__(self):
        super(Net1, self).__init__()
        DROPOUT =0.1
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3,bias = False)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3,bias = False)
        self.conv3 = nn.Conv2d(12, 20, kernel_size=3,bias = False)
        self.conv4 = nn.Conv2d(20, 16, kernel_size=1,bias = False)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        self.conv7 = nn.Conv2d(16, 16, padding = 1, kernel_size=3,bias = False)
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        self.conv9 = nn.Conv2d(16, 10, kernel_size=1,bias = False)
        self.batch1 = nn.BatchNorm2d(12)
        self.batch2 = nn.BatchNorm2d(12)
        self.batch3 = nn.BatchNorm2d(20)
        self.batch4 = nn.BatchNorm2d(16)
        self.batch5 = nn.BatchNorm2d(16)
        self.batch6 = nn.BatchNorm2d(16)
        self.batch7 = nn.BatchNorm2d(16)
        self.batch8 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(DROPOUT)
        self.dropout2 = nn.Dropout2d(DROPOUT)
        # self.dropout3 = nn.Dropout2d(DROPOUT)
        self.dropout4 = nn.Dropout2d(DROPOUT)
        self.dropout5 = nn.Dropout2d(DROPOUT)
        # self.dropout6 = nn.Dropout2d(DROPOUT)
        # self.dropout7 = nn.Dropout2d(DROPOUT)
        self.avgpool = nn.AvgPool2d(5)

    def forward(self, x):
        x = self.dropout1(self.batch1(F.relu(self.conv1(x))))
        x = self.dropout2(self.batch2(F.relu(self.conv2(x))))
        x = self.batch3(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(self.batch4(F.relu(self.conv5(x))))
        x = self.dropout5(self.batch5(F.relu(self.conv6(x))))
        x = self.batch6(F.relu(self.conv7(x)))
        x = self.batch7(F.relu(self.conv8(x)))
        x = self.avgpool(self.conv9(x))
        x = x.view(-1,10)

        return F.log_softmax(x, dim=-1)

    def train_step(self, model, device, train_loader, optimizer):
        # Putting the model to train mode
      model.train()

      # Loading train dtaloader to Tqdm to produce output in bar for
      # visual interpretation.
      correct = 0
      processed = 0
      train_loss_arr = []
      pbar = tqdm(train_loader)
      for batch_idx, (data, target) in enumerate(pbar):
          # adding data and target label to cuda
          data, target = data.to(device), target.to(device)
          # print("\nData Shape:",data.shape)
          # making all the gradients zero before forward propogation
          optimizer.zero_grad()
          # print("Target Shape:",target.shape)
          # loading data to model
          output = model(data)
          # print("output Shape:",output.shape)

          # calculating loss with output and target using negative log likelyhood loss
          loss = F.nll_loss(output, target)
          # train_losses.append(loss)

          # calcualting back propogation
          loss.backward()

          # Revaulating the model and updating the gradient
          optimizer.step()

          pred = output.argmax(dim=1,keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
          processed += len(data)
          train_loss_arr.append(loss)

          pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')
          # train_acc.append(100*correct/processed)

      return 100*correct/processed, train_loss_arr

    def test_step(self, model, device, test_loader):
         # Putting the model to eval mode
        model.eval()

        # Test_loss is kept to 0
        test_loss = 0
        test_loss_arr = []
        # correct value
        correct = 0
        # Loading model without gradient
        with torch.no_grad():
            # Load test model
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # Running model to testing data
                output = model(data)
                # calculating testing call
                temp = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_loss += temp
                # calculating prediction
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # checking all the correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss_arr.append(temp)

        test_loss /= len(test_loader.dataset)
        # test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # test_acc.append(100. * correct / len(test_loader.dataset))

        return 100. * correct / len(test_loader.dataset), test_loss_arr

    def run(self, num_epochs:int, model, device, train_loader, test_loader, optimizer, scheduler):
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            # Training
            temp_acc, temp_loss = model.train_step(model, device, train_loader, optimizer)
            train_acc.append(temp_acc)
            train_losses.extend(temp_loss)
            
            scheduler.step()  # Update the learning rate using the scheduler
            
            # Testing
            temp_acc, temp_loss = model.test_step(model, device, test_loader)
            test_acc.append(temp_acc)
            test_losses.extend(temp_loss)
        return train_acc, train_losses, test_acc, test_losses

class Net3(nn.Module):
    #This defines the structure of the NN.
    
    def __init__(self):
        DROPOUT_VALUE = 0
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3,bias = False)
        self.conv2 = nn.Conv2d(7, 16, kernel_size=3,bias = False)
        # self.conv3 = nn.Conv2d(12, 12, kernel_size=3,bias = False)
        self.conv4 = nn.Conv2d(16, 12, kernel_size=1,bias = False)
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3,bias = False)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        # self.conv8 = nn.Conv2d(11, 11, kernel_size=3,bias = False)
        self.conv9 = nn.Conv2d(16, 10, kernel_size=1,bias = True)
        self.batch1 = nn.BatchNorm2d(7)
        self.batch2 = nn.BatchNorm2d(16)
        self.batch3 = nn.BatchNorm2d(16)
        self.batch4 = nn.BatchNorm2d(16)
        self.batch5 = nn.BatchNorm2d(16)
        self.batch6 = nn.BatchNorm2d(16)
        self.batch7 = nn.BatchNorm2d(16)
        # self.batch8 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout2 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout3 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout4 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout5 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout6 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout7 = nn.Dropout2d(DROPOUT_VALUE)
        self.avgpool = nn.AvgPool2d(5)

    def forward(self, x):
        x = self.dropout1(self.batch1(F.relu(self.conv1(x))))
        x = self.dropout2(self.batch2(F.relu(self.conv2(x))))
        # x = self.dropout3(self.batch3(F.relu(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(self.batch4(F.relu(self.conv5(x))))
        x = self.dropout5(self.batch5(F.relu(self.conv6(x))))
        x = self.dropout6(self.batch6(F.relu(self.conv7(x))))
        # x = self.dropout7(self.batch7(F.relu(self.conv8(x))))
        x = self.avgpool(self.conv9(x))
        x = x.view(-1,10)

        return F.log_softmax(x, dim=-1)

    def train_step(self, model, device, train_loader, optimizer):
        # Putting the model to train mode
      model.train()

      # Loading train dtaloader to Tqdm to produce output in bar for
      # visual interpretation.
      correct = 0
      processed = 0
      train_loss_arr = []
      pbar = tqdm(train_loader)
      for batch_idx, (data, target) in enumerate(pbar):
          # adding data and target label to cuda
          data, target = data.to(device), target.to(device)
          # print("\nData Shape:",data.shape)
          # making all the gradients zero before forward propogation
          optimizer.zero_grad()
          # print("Target Shape:",target.shape)
          # loading data to model
          output = model(data)
          # print("output Shape:",output.shape)

          # calculating loss with output and target using negative log likelyhood loss
          loss = F.nll_loss(output, target)
          # train_losses.append(loss)

          # calcualting back propogation
          loss.backward()

          # Revaulating the model and updating the gradient
          optimizer.step()

          pred = output.argmax(dim=1,keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
          processed += len(data)
          train_loss_arr.append(loss)

          pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')
          # train_acc.append(100*correct/processed)

      return 100*correct/processed, train_loss_arr

    def test_step(self, model, device, test_loader):
         # Putting the model to eval mode
        model.eval()

        # Test_loss is kept to 0
        test_loss = 0
        test_loss_arr = []
        # correct value
        correct = 0
        # Loading model without gradient
        with torch.no_grad():
            # Load test model
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # Running model to testing data
                output = model(data)
                # calculating testing call
                temp = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_loss += temp
                # calculating prediction
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # checking all the correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss_arr.append(temp)

        test_loss /= len(test_loader.dataset)
        # test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # test_acc.append(100. * correct / len(test_loader.dataset))

        return 100. * correct / len(test_loader.dataset), test_loss_arr

    def run(self, num_epochs:int, model, device, train_loader, test_loader, optimizer, scheduler):
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            # Training
            temp_acc, temp_loss = model.train_step(model, device, train_loader, optimizer)
            train_acc.append(temp_acc)
            train_losses.extend(temp_loss)
            
            scheduler.step()  # Update the learning rate using the scheduler
            
            # Testing
            temp_acc, temp_loss = model.test_step(model, device, test_loader)
            test_acc.append(temp_acc)
            test_losses.extend(temp_loss)
        return train_acc, train_losses, test_acc, test_losses
class Net2(nn.Module):
    #This defines the structure of the NN.
    
    def __init__(self):
        DROPOUT_VALUE = 0.1
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3,bias = False)
        self.conv2 = nn.Conv2d(7, 16, kernel_size=3,bias = False)
        # self.conv3 = nn.Conv2d(12, 12, kernel_size=3,bias = False)
        self.conv4 = nn.Conv2d(16, 12, kernel_size=1,bias = False)
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3,bias = False)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3,bias = False)
        # self.conv8 = nn.Conv2d(11, 11, kernel_size=3,bias = False)
        self.conv9 = nn.Conv2d(16, 10, kernel_size=1,bias = False)
        self.batch1 = nn.BatchNorm2d(7)
        self.batch2 = nn.BatchNorm2d(16)
        self.batch3 = nn.BatchNorm2d(16)
        self.batch4 = nn.BatchNorm2d(16)
        self.batch5 = nn.BatchNorm2d(16)
        self.batch6 = nn.BatchNorm2d(16)
        self.batch7 = nn.BatchNorm2d(16)
        # self.batch8 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout2 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout3 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout4 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout5 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout6 = nn.Dropout2d(DROPOUT_VALUE)
        self.dropout7 = nn.Dropout2d(DROPOUT_VALUE)
        self.avgpool = nn.AvgPool2d(5)

    def forward(self, x):
        x = self.dropout1(self.batch1(F.relu(self.conv1(x))))
        x = self.dropout2(self.batch2(F.relu(self.conv2(x))))
        # x = self.dropout3(self.batch3(F.relu(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(self.batch4(F.relu(self.conv5(x))))
        x = self.dropout5(self.batch5(F.relu(self.conv6(x))))
        x = self.dropout6(self.batch6(F.relu(self.conv7(x))))
        # x = self.dropout7(self.batch7(F.relu(self.conv8(x))))
        x = self.avgpool(self.conv9(x))
        x = x.view(-1,10)

        return F.log_softmax(x, dim=-1)

    def train_step(self, model, device, train_loader, optimizer):
        # Putting the model to train mode
      model.train()

      # Loading train dtaloader to Tqdm to produce output in bar for
      # visual interpretation.
      correct = 0
      processed = 0
      train_loss_arr = []
      pbar = tqdm(train_loader)
      for batch_idx, (data, target) in enumerate(pbar):
          # adding data and target label to cuda
          data, target = data.to(device), target.to(device)
          # print("\nData Shape:",data.shape)
          # making all the gradients zero before forward propogation
          optimizer.zero_grad()
          # print("Target Shape:",target.shape)
          # loading data to model
          output = model(data)
          # print("output Shape:",output.shape)

          # calculating loss with output and target using negative log likelyhood loss
          loss = F.nll_loss(output, target)
          # train_losses.append(loss)

          # calcualting back propogation
          loss.backward()

          # Revaulating the model and updating the gradient
          optimizer.step()

          pred = output.argmax(dim=1,keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
          processed += len(data)
          train_loss_arr.append(loss)

          pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')
          # train_acc.append(100*correct/processed)

      return 100*correct/processed, train_loss_arr

    def test_step(self, model, device, test_loader):
         # Putting the model to eval mode
        model.eval()

        # Test_loss is kept to 0
        test_loss = 0
        test_loss_arr = []
        # correct value
        correct = 0
        # Loading model without gradient
        with torch.no_grad():
            # Load test model
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # Running model to testing data
                output = model(data)
                # calculating testing call
                temp = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_loss += temp
                # calculating prediction
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # checking all the correct predictions
                correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss_arr.append(temp)

        test_loss /= len(test_loader.dataset)
        # test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        # test_acc.append(100. * correct / len(test_loader.dataset))

        return 100. * correct / len(test_loader.dataset), test_loss_arr

    def run(self, num_epochs:int, model, device, train_loader, test_loader, optimizer, scheduler):
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        for epoch in range(1, num_epochs+1):
            print(f'Epoch {epoch}')
            # Training
            temp_acc, temp_loss = model.train_step(model, device, train_loader, optimizer)
            train_acc.append(temp_acc)
            train_losses.extend(temp_loss)
            
            scheduler.step()  # Update the learning rate using the scheduler
            
            # Testing
            temp_acc, temp_loss = model.test_step(model, device, test_loader)
            test_acc.append(temp_acc)
            test_losses.extend(temp_loss)
        return train_acc, train_losses, test_acc, test_losses
