import torch
import copy, time


__all__ = ['TrainHandler', 'GDTrainHandler']


class TrainHandler():
    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device='cuda:0'):
        self.model = model.to(device)
        self.dataloaders, self.dataset_sizes = dataloaders, dataset_sizes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def _epoch_phase(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss, running_correct = 0.0, 0.0

        for inputs, labels in self.dataloaders[phase]:
            inputs, labels = inputs.to(self.device), labels.to(self.device)           

            # Zero out parameter gradients
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                # Forward pass
                outputs = self.model(inputs)
                preds = outputs.max(dim=1)[1]
                loss = self.criterion(outputs, labels)

                # Backward pass
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data).item()

        if (phase == 'train') and (self.scheduler != None):
            self.scheduler.step()

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc  = running_correct / self.dataset_sizes[phase]

        return round(epoch_loss, 4), round(epoch_acc, 4)


    def train_model(self, num_epochs=100, test_freq=1, print_freq=1):
        since = time.time()
        train_losses, test_losses, train_accs, test_accs = [], [], [], []
        self.num_epochs = num_epochs

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(1, self.num_epochs + 1):
            # Each epoch have training and testphases
            epoch_start = time.time()
            train_loss, train_acc = self._epoch_phase('train')
            if epoch % test_freq == 0:
                test_loss, test_acc = self._epoch_phase('test')

            epoch_elapse = round(time.time() - epoch_start, 3)

            # Save loss and accuracy statistics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            if epoch % test_freq == 0:
                test_losses.append((epoch, test_loss))
                test_accs.append((epoch, test_acc))

            # Update the best test accuracy
            if test_acc > best_acc or self.dataset_sizes['test'] == 0:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
            if epoch % print_freq == 0:
                self._print_train_stat(epoch, num_epochs, epoch_elapse, train_loss, train_acc, test_loss, test_acc)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
        print('=' * 50, '\n')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        return train_losses, test_losses, train_accs, test_accs
    
    def _print_train_stat(self, epoch, num_epochs, epoch_elapse, train_loss, train_acc, test_loss, test_acc):
        print('[Epoch {}/{}] Elapsed {}s/it'.format(epoch, num_epochs, epoch_elapse))

        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
        
        # train_acc and test_acc are just a scalar value
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%, Learning Rate - {:0.6f}'.format('Train', train_loss, train_acc * 100, learning_rate))
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%'.format('Test', test_loss, test_acc * 100))
        print('======================================================================')


class GDTrainHandler():
    def __init__(self, models, dataloaders, dataset_sizes, criterion, optimizers, scheduler=None, device='cuda:0'):
        self.model_g, self.model_d = models[0].to(device), models[1].to(device)
        self.dataloaders, self.dataset_sizes = dataloaders, dataset_sizes
        self.criterion = criterion
        self.optimizer_g, self.optimizer_d = optimizers[0], optimizers[1]
        self.scheduler = scheduler
        self.device = device
        
    def _epoch_phase(self, phase):
        if phase == 'train':
            self.model_g.train()
            self.model_d.train()
        else:
            self.model_g.eval()
            self.model_d.eval()

        running_loss_g, running_loss_d = 0.0, 0.0

        for images, labels in self.dataloaders[phase]:
            images, labels = images.to(self.device), labels.to(self.device)           
            
            y_real = torch.ones(images.size(0), 1).to(self.device)
            y_fake = torch.zeros(images.size(0), 1).to(self.device)
            
            ## Train Discriminator ---------------------
            with torch.set_grad_enabled(phase == 'train'):

                # on real data
                D_real_decision = self.model_d(images, labels)
                D_real_loss = self.criterion(D_real_decision, y_real)

                # on fake data
                noise = torch.randn(images.size(0), 100).view(-1, 100, 1, 1).to(self.device)
                rand_labels = (torch.rand(images.size(0)) * 20).long().squeeze().to(self.device)
                fake_imgs = self.model_g(noise, rand_labels)

                D_fake_decision = self.model_d(fake_imgs, rand_labels)
                D_fake_loss = self.criterion(D_fake_decision, y_fake)

                # Discriminator Backward Pass
                D_loss = D_real_loss + D_fake_loss
                
                if phase == 'train':
                    self.optimizer_d.zero_grad()
                    D_loss.backward()
                    self.optimizer_d.step()


                ## Train Generator ---------------------
                noise = torch.randn(images.size(0), 100).view(-1, 100, 1, 1).to(self.device)
                rand_labels = (torch.rand(images.size(0)) * 20).long().squeeze().to(self.device)
                fake_imgs = self.model_g(noise, rand_labels)

                D_fake_decision = self.model_d(fake_imgs, rand_labels).view(fake_imgs.size(0), 1)
                G_loss = self.criterion(D_fake_decision, y_real)
                
                # Generator Backward Pass
                if phase == 'train':
                    self.optimizer_g.zero_grad()
                    G_loss.backward()
                    self.optimizer_g.zero_grad()

            # Statistics
            running_loss_g += G_loss.item() * images.size(0)
            running_loss_d += D_loss.item() * images.size(0)
            
        if (phase == 'train') and (self.scheduler != None):
            self.scheduler.step()

        epoch_loss_g = running_loss_g / self.dataset_sizes[phase]
        epoch_loss_d = running_loss_g / self.dataset_sizes[phase]

        return round(epoch_loss_g, 4), round(epoch_loss_d, 4)


    def train_model(self, num_epochs=100, test_freq=1, print_freq=1):
        since = time.time()
        g_losses_train, d_losses_train = [], []
        g_losses_test, d_losses_test = [], []
        
        self.num_epochs = num_epochs

        for epoch in range(1, self.num_epochs + 1):
            # Each epoch have training and testphases
            epoch_start = time.time()
            loss_g_train, loss_d_train = self._epoch_phase('train')
            
            if epoch % test_freq == 0:
                loss_g_test, loss_d_test = self._epoch_phase('test')

            epoch_elapse = round(time.time() - epoch_start, 3)

            # Save loss and accuracy statistics
            g_losses_train.append(loss_g_train)
            d_losses_train.append(loss_d_train)
            
            if epoch % test_freq == 0:
                g_losses_test.append((epoch, loss_g_test))
                d_losses_test.append((epoch, loss_d_test))

            if epoch % print_freq == 0:
                self._print_train_stat(epoch, num_epochs, epoch_elapse, 
                                       loss_g_train, loss_d_train, 
                                       loss_g_test, loss_d_test)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('=' * 50, '\n')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        return train_losses, test_losses, train_accs, test_accs
    
    def _print_train_stat(self, epoch, num_epochs, epoch_elapse, loss_g_train, loss_d_train, loss_g_test, loss_d_test):
        print('[Epoch {}/{}] Elapsed {}s/it'.format(epoch, num_epochs, epoch_elapse))

        for param_group in self.optimizer_g.param_groups:
            learning_rate = param_group['lr']
        
        # train_acc and test_acc are just a scalar value
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%, Learning Rate - {:0.6f}'.format('Train', loss_g_train, loss_d_train, learning_rate))
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%'.format('Test', loss_g_test, loss_d_test))
        print('======================================================================')
