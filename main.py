import os 
import random 
from string import ascii_letters

import torch 
from torch import nn
import torch.nn.functional as F

from unidecode import unidecode 
from sklearn.model_selection import train_test_split
from rnn import CustomRNN
from gru import CustomGRU

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "data/names"
lang2labels = { file_name.split('.')[0]:torch.tensor([i], dtype=torch.long) \
               for i, file_name in enumerate(os.listdir(data_dir))}

num_langs = len(lang2labels)

unidecode('Ślusàrski') # -> output: Slusarski remove strange chars
char2idx = {letter: i for i, letter in enumerate(ascii_letters + " .,:;-'")}
num_letters = len(char2idx)
num_letters

def name2tensor(name: str):
    tensor = torch.zeros(len(name), 1, num_letters)
    for i, ch in enumerate(name):
        tensor[i][0][char2idx[ch]] = 1
    return tensor

print(name2tensor("abc"))

tensor_names = []
target_langs = []

for file_ in os.listdir(data_dir):
    with open(os.path.join(data_dir, file_)) as f:
        lang = file_.split('.')[0]
        names = [unidecode(l.strip()) for l in f]
        for name in names:
            try:
                tensor_names.append(name2tensor(name))
                target_langs.append(lang2labels[lang])
            except KeyError:
                pass

train_idx, test_idx = train_test_split(
    range(len(target_langs)), 
    test_size=0.1, 
    shuffle=True, 
    stratify=target_langs
)
train_dataset = [
    (tensor_names[i], target_langs[i])
    for i in train_idx
]

test_dataset = [
    (tensor_names[i], target_langs[i])
    for i in test_idx 
]

print(f"Train: {len(train_dataset)}")
print(f"Test: {len(test_dataset)}")


hidden_size = 256
lr = 0.0001

model = CustomRNN(num_letters, hidden_size, num_langs)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 5 
print_interval = 3000
for epoch in range(epochs):
    random.shuffle(train_dataset)
    for i,(name, label) in enumerate(train_dataset):
        hidden_state = model.init_hidden()
        for char in name:
            output, hidden_state = model(char, hidden_state)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        if (i + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Step [{i + 1}/{len(train_dataset)}], "
                f"Loss: {loss.item():.4f}"
            )
            
num_correct = 0
num_samples = len(test_dataset)
model.eval()
with torch.no_grad():
    for name, label in test_dataset:
        hidden_state = model.init_hidden()
        for char in name:
            output, hidden = model(char, hidden_state)
        _, pred = torch.max(output, 1)
        num_correct += bool(pred == label)
print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")


hidden_size = 256
lr = 0.001

model = CustomGRU(num_letters, hidden_size, num_langs)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2 
print_interval = 3000
for epoch in range(epochs):
    random.shuffle(train_dataset)
    for i,(name, label) in enumerate(train_dataset):
        hidden_state = model.init_hidden()
        for char in name:
            output, hidden_state = model(char, hidden_state)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        if (i + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Step [{i + 1}/{len(train_dataset)}], "
                f"Loss: {loss.item():.4f}"
            )
            
num_correct = 0
num_samples = len(test_dataset)
model.eval()
with torch.no_grad():
    for name, label in test_dataset:
        hidden_state = model.init_hidden()
        for char in name:
            output, hidden = model(char, hidden_state)
        _, pred = torch.max(output, 1)
        num_correct += bool(pred == label)

print(f"Accuracy: {num_correct / num_samples * 100:.4f}%")