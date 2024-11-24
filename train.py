import torch 
from tqdm import tqdm

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(n_epochs)):

        model.train()
        train_loss = 0
        for idx, (img, mask) in enumerate(tqdm(train_loader)):
            img = img.float().to(device)
            mask = mask.float().to(device)
            
            y_pred = model(img)
            loss = loss_fn(y_pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / (idx+1)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(val_loader)):
                img = img.float().to(device)
                mask = mask.float().to(device)
                y_pred = model(img)
                loss = loss_fn(y_pred, mask)
                val_loss += loss.item()
            
            val_loss = val_loss / (idx+1)
            val_losses.append(val_loss)
        
        print("-"*30)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}")
        print(f"Epoch: {epoch+1}, Val Loss: {val_loss:.4f}")
        print("-"*30)
    
    return train_losses, val_losses