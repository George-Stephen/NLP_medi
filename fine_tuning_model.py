import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import string

# Load the medical questions and answers from CSV
data = pd.read_csv('medical_qa_data.csv')

# Preprocess the data
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['Answer'] = data['Answer'].apply(preprocess_text)

# Tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

# Prepare dataset
class MedicalQADataset(Dataset):
    def __init__(self, questions, contexts, answers):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.encodings = tokenizer(contexts, questions, truncation=True, padding=True, return_tensors='pt')
        self.add_token_positions()

    def add_token_positions(self):
        start_positions = []
        end_positions = []
        for i in range(len(self.answers)):
            answer = self.answers[i]
            context = self.contexts[i]
            start_idx = context.find(answer)
            if start_idx == -1:
                start_idx = 0  # If answer not found in context, set to 0 to avoid errors
            end_idx = start_idx + len(answer)
            encodings = tokenizer(context, return_offsets_mapping=True, truncation=True, padding=True)
            offset_mapping = encodings['offset_mapping'][0]

            # Find start and end positions
            start_positions.append(self.get_position(offset_mapping, start_idx))
            end_positions.append(self.get_position(offset_mapping, end_idx - 1))

        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    def get_position(self, offset_mapping, char_position):
        for i, (start, end) in enumerate(offset_mapping):
            if start <= char_position < end:
                return i
        return len(offset_mapping) - 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items() if key in ['input_ids', 'attention_mask', 'start_positions', 'end_positions']}
        return item

questions = data['Question'].tolist()
contexts = data['Answer'].tolist()
answers = data['Answer'].tolist()  # Assuming the answer is within the context

dataset = MedicalQADataset(questions, contexts, answers)

# Split data into train and test
train_size = 0.8
train_dataset, val_dataset = train_test_split(dataset, train_size=train_size, random_state=42)

# DataLoader for the datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=50,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_qa_model')
tokenizer.save_pretrained('fine_tuned_qa_model')
