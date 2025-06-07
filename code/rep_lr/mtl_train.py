import os
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn

def train_mtl_model(models, train_dl, val_dl, loss_fns, optimizer, args):
    """
    Main training loop for Multi-Task Learning.
    """
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fusion_model = models['fusion'].to(device)
    rf_model = models['rf'].to(device)
    channel_model = models['channel'].to(device)
    cfo_model = models['cfo'].to(device)

    wandb.watch((fusion_model, rf_model, channel_model, cfo_model), log='all', log_freq=100)

    best_val_loss = float('inf')
    patience_counter = 0

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for epoch in range(args.epochs):
        fusion_model.train()
        rf_model.train()
        channel_model.train()
        cfo_model.train()
        
        total_train_loss = 0.0
        train_losses = {task: 0.0 for task in ['rf', 'channel', 'cfo']}
        train_correct_rf = 0
        train_total_rf = 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y = [d.to(device, non_blocking=True) for d in batch]
            
            rf_x, cfo_x, channel_x = rf_x.float(), cfo_x.float(), channel_x.float()
            rf_y = rf_y.long()
            cfo_y, channel_y = cfo_y.float(), channel_y.float()

            optimizer.zero_grad()
            
            common_rep = fusion_model(rf_x, cfo_x, channel_x)
            
            # Task-specific heads
            rf_out = rf_model(rf_x, common_rep)
            channel_out = channel_model(channel_x, common_rep)
            cfo_out = cfo_model(cfo_x, common_rep)

            # Losses
            loss_rf = loss_fns['rf'](rf_out, rf_y)
            loss_channel = loss_fns['channel'](channel_out, channel_y)
            loss_cfo = loss_fns['cfo'](cfo_out, cfo_y)

            # Combine losses (with weights if specified, else equal weighting)
            total_loss = args.loss_weight_rf * loss_rf + \
                         args.loss_weight_channel * loss_channel + \
                         args.loss_weight_cfo * loss_cfo
            
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            train_losses['rf'] += loss_rf.item()
            train_losses['channel'] += loss_channel.item()
            train_losses['cfo'] += loss_cfo.item()

            _, predicted = torch.max(rf_out.data, 1)
            train_total_rf += rf_y.size(0)
            train_correct_rf += (predicted == rf_y).sum().item()

        avg_train_loss = total_train_loss / len(train_dl)
        avg_train_losses = {task: loss / len(train_dl) for task, loss in train_losses.items()}
        train_acc_rf = 100 * train_correct_rf / train_total_rf
        
        log_dict = {
            'train/total_loss': avg_train_loss,
            'train/rf_loss': avg_train_losses['rf'],
            'train/channel_loss': avg_train_losses['channel'],
            'train/cfo_loss': avg_train_losses['cfo'],
            'train/rf_accuracy': train_acc_rf
        }
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, RF Acc: {train_acc_rf:.2f}%")

        # Validation
        fusion_model.eval()
        rf_model.eval()
        channel_model.eval()
        cfo_model.eval()

        total_val_loss = 0.0
        val_losses = {task: 0.0 for task in ['rf', 'channel', 'cfo']}
        val_correct_rf = 0
        val_total_rf = 0

        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                rf_x, rf_y, cfo_x, cfo_y, channel_x, channel_y = [d.to(device, non_blocking=True) for d in batch]
                
                rf_x, cfo_x, channel_x = rf_x.float(), cfo_x.float(), channel_x.float()
                rf_y = rf_y.long()
                cfo_y, channel_y = cfo_y.float(), channel_y.float()
                
                common_rep = fusion_model(rf_x, cfo_x, channel_x)
            
                rf_out = rf_model(rf_x, common_rep)
                channel_out = channel_model(channel_x, common_rep)
                cfo_out = cfo_model(cfo_x, common_rep)

                loss_rf = loss_fns['rf'](rf_out, rf_y)
                loss_channel = loss_fns['channel'](channel_out, channel_y)
                loss_cfo = loss_fns['cfo'](cfo_out, cfo_y)

                total_loss = args.loss_weight_rf * loss_rf + \
                             args.loss_weight_channel * loss_channel + \
                             args.loss_weight_cfo * loss_cfo

                total_val_loss += total_loss.item()
                val_losses['rf'] += loss_rf.item()
                val_losses['channel'] += loss_channel.item()
                val_losses['cfo'] += loss_cfo.item()

                _, predicted = torch.max(rf_out.data, 1)
                val_total_rf += rf_y.size(0)
                val_correct_rf += (predicted == rf_y).sum().item()

        avg_val_loss = total_val_loss / len(val_dl)
        avg_val_losses = {task: loss / len(val_dl) for task, loss in val_losses.items()}
        val_acc_rf = 100 * val_correct_rf / val_total_rf
        
        log_dict.update({
            'val/total_loss': avg_val_loss,
            'val/rf_loss': avg_val_losses['rf'],
            'val/channel_loss': avg_val_losses['channel'],
            'val/cfo_loss': avg_val_losses['cfo'],
            'val/rf_accuracy': val_acc_rf
        })
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val RF Acc: {val_acc_rf:.2f}%")
        
        wandb.log(log_dict, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'fusion_model': fusion_model.state_dict(),
                'rf_model': rf_model.state_dict(),
                'channel_model': channel_model.state_dict(),
                'cfo_model': cfo_model.state_dict(),
            }, os.path.join(args.save_path, 'mtl_best.pt'))
            print("Saved best models.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            torch.save({
                'fusion_model': fusion_model.state_dict(),
                'rf_model': rf_model.state_dict(),
                'channel_model': channel_model.state_dict(),
                'cfo_model': cfo_model.state_dict(),
            }, os.path.join(args.save_path, f'mtl_epoch_{epoch+1}.pt'))
            print(f"Saved checkpoint at epoch {epoch+1}")

        if patience_counter >= args.patience:
            print("Early stopping.")
            break 