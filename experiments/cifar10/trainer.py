# torch
import torch
# built-in
import numpy as np
import copy


def train(model, dataloaders, args):
    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # create optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # Get lr scheduler
    if 'allcnnc' in args.model:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)
    elif 'resnet' in args.model:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    else: lr_scheduler = None

    # train network
    _, va, vl, ta, tl = _train(model, args.epochs, criterion, optimizer, dataloaders, args.device, lr_scheduler)
    # save model
    torch.save(model.state_dict(), args.path)
    # save history
    history = np.array([va, vl, ta, tl])
    np.save(args.path[:-4] + "_history.npy", history)


def _train(model, epochs, criterion, optimizer, dataloader, device, lr_scheduler):
    # Accumulate information about the training history
    val_acc_history = []
    train_acc_history = []
    loss_train_history = []
    loss_val_history = []
    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # iterate over epochs
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 30)
        # Print current learning rate
        if lr_scheduler is not None:
           for param_group in optimizer.param_groups:
               print('Learning Rate: {}'.format(param_group['lr']))
           print('-' * 30)

        # Each epoch consist of training and validation
        for phase in ['train', 'validation']:
            if phase == 'train': model.train()
            else: model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # FwrdPhase:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            import datetime
            print(datetime.datetime.now())

            # Store results
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                loss_train_history.append(epoch_loss)
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                loss_val_history.append(epoch_loss)

            # If better validation accuracy, replace best weights
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Update scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        print()

    # Report best results
    print('Best Val Acc: {:.4f}'.format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Return model and histories
    return model, val_acc_history, loss_val_history, train_acc_history, loss_train_history
