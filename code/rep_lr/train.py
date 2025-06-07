import os
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn


def train_model(model, train_dl, val_dl, loss_fn, optimizer, args):
    """
    Main training loop.
    """
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Watch model with wandb
    wandb.watch(model, log='all', log_freq=100)

    best_val_loss = float('inf')
    patience_counter = 0

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Data indices based on task
    if args.task == 'rf_fingerprinting':
        data_indices = (0, 1)  # RF_X, RF_y
    elif args.task == 'channel_estimation':
        data_indices = (4, 5)  # Channel_X, Channel_y
    elif args.task == 'cfo_estimation_small':
        data_indices = (2, 3)  # CFO_X, CFO_y
    elif args.task == 'cfo_estimation_large':
        data_indices = (2, 3)  # CFO_X, CFO_y
    else:
        raise ValueError(f"Unknown task: {args.task}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs = batch[data_indices[0]].to(device, non_blocking=True).float()
            labels = batch[data_indices[1]].to(device, non_blocking=True)

            if args.task == 'rf_fingerprinting':
                labels = labels.long()
            else:
                labels = labels.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if args.task == 'rf_fingerprinting':
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_dl)
        log_dict = {'train/loss': avg_train_loss}
        if args.task == 'rf_fingerprinting':
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            log_dict['train/accuracy'] = train_acc
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs = batch[data_indices[0]].to(device, non_blocking=True).float()
                labels = batch[data_indices[1]].to(device, non_blocking=True)
                
                if args.task == 'rf_fingerprinting':
                    labels = labels.long()
                else:
                    labels = labels.float()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                if args.task == 'rf_fingerprinting':
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_dl)
        log_dict['val/loss'] = avg_val_loss
        if args.task == 'rf_fingerprinting':
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            log_dict['val/accuracy'] = val_acc
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        wandb.log(log_dict, step=epoch)

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.task}_best.pt'))
            print("Saved best model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.task}_epoch_{epoch+1}.pt'))
            print(f"Saved checkpoint at epoch {epoch+1}")

        if patience_counter >= args.patience:
            print("Early stopping.")
            break 