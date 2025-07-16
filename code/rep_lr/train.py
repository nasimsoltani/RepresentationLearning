import os
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


def train_model(model, train_dl, val_dl, loss_fn, optimizer, args):
    """
    Main training loop.
    """
    try:
        # Set device
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Initialize scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        # Watch model with wandb
        try:
            wandb.watch(model, log='all', log_freq=100)
        except Exception as e:
            print(f"Warning: Failed to initialize wandb watching: {str(e)}")

        best_val_loss = float('inf')
        patience_counter = 0

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # Data indices based on task
        TASK_TO_INDICES = {
            'rf_fingerprinting': (0, 1),  # RF_X, RF_y
            'cfo_estimation': (2, 3),  # CFO_X, CFO_y
            'channel_estimation': (4, 5)  # Channel_X, Channel_y
        }
        
        TASK_TO_LOSS_WEIGHT = {
            'rf_fingerprinting': args.w_rf,
            'cfo_estimation': args.w_cfo,
            'channel_estimation': args.w_channel
        }

        for epoch in range(args.epochs):
            # ===================================
            #           TRAINING PHASE
            # ===================================
            model.train()
            
            if args.mtl:
                # ================== MTL Training ==================
                train_losses = {task: 0.0 for task in args.task}
                train_correct = {task: 0 for task in args.task}
                train_total = {task: 0 for task in args.task}
                
                for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
                    optimizer.zero_grad()
                    
                    # Projections
                    projected_tensors = []
                    for task in args.task:
                        data_idx, _ = TASK_TO_INDICES[task]
                        inputs = batch[data_idx].to(device, non_blocking=True).float()
                        projected_tensors.append(model['projections'][task](inputs))
                    
                    projected_sum = torch.sum(torch.stack(projected_tensors), dim=0)
                    
                    # Shared Encoder
                    encoded = model['encoder'](projected_sum)
                    
                    # Task Heads and Loss Calculation
                    total_loss = 0
                    for task in args.task:
                        _, label_idx = TASK_TO_INDICES[task]
                        labels = batch[label_idx].to(device, non_blocking=True)
                        if task == 'rf_fingerprinting':
                            labels = labels.long()
                        else:
                            labels = labels.float()

                        
                        
                        outputs = model['heads'][task](encoded)
                        task_loss = loss_fn[task](outputs, labels)
                        
                        if torch.isnan(task_loss) or not isinstance(task_loss.item(), float):
                            print(f"Warning: NaN loss for task {task} in batch {batch_idx}, skipping...")
                            continue
                        
                        weight = TASK_TO_LOSS_WEIGHT[task]
                        total_loss += weight * task_loss
                        
                        train_losses[task] += task_loss.item()
                        
                        if task == 'rf_fingerprinting':
                            _, predicted = torch.max(outputs.data, 1)
                            train_total[task] += labels.size(0)
                            train_correct[task] += (predicted == labels).sum().item()

                    if total_loss == 0: continue

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                # Calculate and Log Training Metrics for MTL
                avg_train_losses = {task: loss / len(train_dl) for task, loss in train_losses.items()}
                log_dict = {'train/learning_rate': optimizer.param_groups[0]['lr']}
                print(f"\nEpoch {epoch+1} Train Metrics:")
                for task in args.task:
                    log_dict[f'train/loss_{task}'] = avg_train_losses[task]
                    print(f"  {task} Loss: {avg_train_losses[task]:.4f}", end="")
                    if task == 'rf_fingerprinting':
                        acc = 100 * train_correct[task] / train_total[task] if train_total[task] > 0 else 0
                        log_dict['train/accuracy_rf_fingerprinting'] = acc
                        print(f", Acc: {acc:.2f}%")
                    else:
                        print()
                print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            else:
                # ================== Single-Task Training ==================
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                data_indices = TASK_TO_INDICES[args.task]

                for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")):
                    try:
                        inputs = batch[data_indices[0]].to(device, non_blocking=True).float()
                        labels = batch[data_indices[1]].to(device, non_blocking=True)

                        if args.task == 'rf_fingerprinting':
                            labels = labels.long()
                        else:
                            labels = labels.float()

                        optimizer.zero_grad()

                        projection, encoder, task_head = model
                        x = projection(inputs)
                        x = encoder(x)
                        outputs = task_head(x)

                        loss = loss_fn(outputs, labels)
                        
                        if torch.isnan(loss) or not isinstance(loss.item(), float):
                            print(f"Warning: NaN loss detected in batch {batch_idx}, skipping...")
                            continue

                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()

                        train_loss += loss.item()

                        #Log average of Y_true to wandb for debugging
                        wandb.log({'train/y_true_mean': torch.mean(labels).item()}, step=epoch)
                        #Log norm of gradients to wandb for debugging
                        wandb.log({'train/grad_norm': torch.norm(torch.stack([p.grad.norm() for p in model.parameters()])).item()}, step=epoch)

                        if args.task == 'rf_fingerprinting':
                            _, predicted = torch.max(outputs.data, 1)
                            train_total += labels.size(0)
                            train_correct += (predicted == labels).sum().item()

                    except Exception as e:
                        print(f"Warning: Error in training batch {batch_idx}: {str(e)}")
                        continue

                # Calculate and Log Training Metrics for Single Task
                avg_train_loss = train_loss / len(train_dl)
                log_dict = {
                    'train/loss': avg_train_loss,
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                
                if args.task == 'rf_fingerprinting':
                    train_acc = 100 * train_correct / train_total if train_total > 0 else 0
                    log_dict['train/accuracy'] = train_acc
                    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

           
            # ===================================
            #          VALIDATION PHASE
            # ===================================
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                if args.mtl:
                    # ================== MTL Validation ==================
                    val_losses = {task: 0.0 for task in args.task}
                    val_correct = {task: 0 for task in args.task}
                    val_total = {task: 0 for task in args.task}

                    for batch_idx, batch in enumerate(tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")):
                        # Projections
                        projected_tensors = []
                        for task in args.task:
                            data_idx, _ = TASK_TO_INDICES[task]
                            inputs = batch[data_idx].to(device, non_blocking=True).float()
                            projected_tensors.append(model['projections'][task](inputs))

                        projected_sum = torch.sum(torch.stack(projected_tensors), dim=0)
                        
                        # Shared Encoder
                        encoded = model['encoder'](projected_sum)
                        
                        # Task Heads and Loss
                        for task in args.task:
                            _, label_idx = TASK_TO_INDICES[task]
                            labels = batch[label_idx].to(device, non_blocking=True)
                            if task == 'rf_fingerprinting':
                                labels = labels.long()
                            else:
                                labels = labels.float()
                            
                            outputs = model['heads'][task](encoded)
                            task_loss = loss_fn[task](outputs, labels)
                            
                            if torch.isnan(task_loss) or not isinstance(task_loss.item(), float): continue
                            
                            val_losses[task] += task_loss.item()
                            
                            if task == 'rf_fingerprinting':
                                _, predicted = torch.max(outputs.data, 1)
                                val_total[task] += labels.size(0)
                                val_correct[task] += (predicted == labels).sum().item()

                    # Calculate and Log Validation Metrics for MTL
                    avg_val_losses = {task: loss / len(val_dl) for task, loss in val_losses.items()}
                    print(f"Epoch {epoch+1} Val Metrics:")
                    for task in args.task:
                        log_dict[f'val/loss_{task}'] = avg_val_losses[task]
                        total_val_loss += avg_val_losses[task] # For scheduler and best model
                        print(f"  {task} Val Loss: {avg_val_losses[task]:.4f}", end="")
                        if task == 'rf_fingerprinting':
                            acc = 100 * val_correct[task] / val_total[task] if val_total[task] > 0 else 0
                            log_dict['val/accuracy_rf_fingerprinting'] = acc
                            print(f", Val Acc: {acc:.2f}%")
                        else:
                            print()
                    log_dict['val/total_loss'] = total_val_loss

                else:
                    # ================== Single-Task Validation ==================
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    data_indices = TASK_TO_INDICES[args.task]

                    for batch_idx, batch in enumerate(tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")):
                        try:
                            inputs = batch[data_indices[0]].to(device, non_blocking=True).float()
                            labels = batch[data_indices[1]].to(device, non_blocking=True)
                            
                            if args.task == 'rf_fingerprinting':
                                labels = labels.long()
                            else:
                                labels = labels.float()

                            projection, encoder, task_head = model
                            x = projection(inputs)
                            x = encoder(x)
                            outputs = task_head(x)

                            loss = loss_fn(outputs, labels)
                            
                            if torch.isnan(loss) or not isinstance(loss.item(), float): continue
                                
                            val_loss += loss.item()

                            if args.task == 'rf_fingerprinting':
                                _, predicted = torch.max(outputs.data, 1)
                                val_total += labels.size(0)
                                val_correct += (predicted == labels).sum().item()

                        except Exception as e:
                            print(f"Warning: Error in validation batch {batch_idx}: {str(e)}")
                            continue
                    
                    # Calculate and Log Validation Metrics for Single Task
                    total_val_loss = val_loss / len(val_dl)
                    log_dict['val/loss'] = total_val_loss
                    if args.task == 'rf_fingerprinting':
                        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                        log_dict['val/accuracy'] = val_acc
                        print(f"Epoch {epoch+1} Val Loss: {total_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                    else:
                        print(f"Epoch {epoch+1} Val Loss: {total_val_loss:.4f}")

            # Step scheduler, log to wandb, and save model
            scheduler.step()
            try:
                wandb.log(log_dict, step=epoch)
            except Exception as e:
                print(f"Warning: Failed to log metrics to wandb: {str(e)}")

            # Save checkpoints
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                patience_counter = 0
                
                checkpoint = {'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best_val_loss}
                
                if args.mtl:
                    checkpoint['projections_state_dict'] = model['projections'].state_dict()
                    checkpoint['encoder_state_dict'] = model['encoder'].state_dict()
                    checkpoint['heads_state_dict'] = model['heads'].state_dict()
                else:
                    for i, module in enumerate(model):
                        checkpoint[f'module_{i}'] = module.state_dict()
                
                save_path = os.path.join(args.save_path, f"{'_'.join(args.task) if args.mtl else args.task}_best.pt")
                torch.save(checkpoint, save_path)
                print("Saved best model.")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

            if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
                checkpoint = {'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'val_loss': total_val_loss}
                
                if args.mtl:
                    checkpoint['projections_state_dict'] = model['projections'].state_dict()
                    checkpoint['encoder_state_dict'] = model['encoder'].state_dict()
                    checkpoint['heads_state_dict'] = model['heads'].state_dict()
                else:
                    for i, module in enumerate(model):
                        checkpoint[f'module_{i}'] = module.state_dict()
                
                save_path = os.path.join(args.save_path, f"{'_'.join(args.task) if args.mtl else args.task}_epoch_{epoch+1}.pt")
                torch.save(checkpoint, save_path)
                print(f"Saved checkpoint at epoch {epoch+1}")

            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    except Exception as e:
        print(f"Critical error in training loop: {str(e)}")
        # Try to save emergency checkpoint
        try:
            checkpoint = {'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
            
            if args.mtl:
                checkpoint['projections_state_dict'] = model['projections'].state_dict()
                checkpoint['encoder_state_dict'] = model['encoder'].state_dict()
                checkpoint['heads_state_dict'] = model['heads'].state_dict()
            elif isinstance(model, nn.ModuleList):
                for i, module in enumerate(model):
                    checkpoint[f'module_{i}'] = module.state_dict()
            else: # Fallback for unexpected model types
                checkpoint['model_state_dict'] = model.state_dict()
            
            save_path = os.path.join(args.save_path, f"{'_'.join(args.task) if args.mtl else args.task}_emergency.pt")
            torch.save(checkpoint, save_path)
            print("Saved emergency checkpoint.")
        except Exception as save_error:
            print(f"Failed to save emergency checkpoint: {str(save_error)}") 