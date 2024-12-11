import torch
from torch.utils.data import DataLoader

def evaluate_model(model, eval_dataset, collate_fn, batch_size=1, device="cpu"):
    model.to(device)
    model.eval()

    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            n_tweets = batch['n_tweets'].to(device)
            n_words = batch['n_words'].to(device)

            outputs = model(input_ids=input_ids, n_tweets=n_tweets, n_words=n_words)
            probs = outputs['logits'].cpu().numpy()

            # Convert probabilities to binary predictions
            preds = (probs >= 0.5).astype(int)

            # Retrieve original IDs and labels (you may need to adjust this if IDs are not in eval_dataset)
            
            results.extend(preds)

    return results