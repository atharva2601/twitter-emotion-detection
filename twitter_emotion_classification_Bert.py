import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import classification_report

# Load datasets
training_data = pd.read_csv('Dataset/training.csv')
validation_data = pd.read_csv('Dataset/validation.csv')
test_data = pd.read_csv('Dataset/test.csv')

# Prepare tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = training_data['label'].nunique()

# Tokenize data
def tokenize_data(data, labels):
    encoding = tokenizer(data.tolist(), max_length=64, truncation=True, padding="max_length", return_tensors="pt")
    dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'], torch.tensor(labels.tolist()))
    return dataset

train_dataset = tokenize_data(training_data['text'], training_data['label'])
val_dataset = tokenize_data(validation_data['text'], validation_data['label'])
test_dataset = tokenize_data(test_data['text'], test_data['label'])

# For fast dev, reduce training data size
train_size = int(0.1 * len(train_dataset))
small_train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

# Loaders
train_loader = DataLoader(small_train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch_input_ids, batch_attention_mask, batch_labels = batch
        model.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Avg training loss Epoch {epoch+1}: {total_loss / len(train_loader):.4f}")

    model.eval()
    total_eval_loss = 0
    for batch in val_loader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        total_eval_loss += outputs.loss.item()
    print(f"Validation loss Epoch {epoch+1}: {total_eval_loss / len(val_loader):.4f}")

# Save the model
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Evaluate
def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
            true_labels.extend(batch_labels.cpu().numpy())
    return classification_report(true_labels, predictions, target_names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])

report = evaluate_model(model, test_loader)
print(report)