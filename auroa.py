import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import nn
import numpy as np
from pathlib import Path
import logging
import yaml
import json
from typing import Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
import argparse
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QTextEdit, QLineEdit, QLabel, QComboBox, QProgressBar,
                            QFileDialog, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys
import ollama
from dataset_generator import generate_training_data

# Configuration Class
class Config:
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    MAX_LENGTH = 512
    MODEL_CHECKPOINT = "bert-base-uncased"
    SAVE_DIR = Path("models/aurora")
    LOG_DIR = Path("logs")
    DATA_DIR = Path("data")
    VALIDATION_SPLIT = 0.2
    SEED = 42
    OLLAMA_MODEL = "llama3"

# Ensure directories exist before logging setup
for directory in [Config.LOG_DIR, Config.SAVE_DIR, Config.DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_DIR / "aurora_training.log"),
        logging.StreamHandler()
    ]
)

# Training Worker Thread
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model, train_loader, val_loader, device, epochs):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

    def run(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        total_steps = len(self.train_loader) * self.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=Config.LEARNING_RATE, total_steps=total_steps, pct_start=0.3
        )
        best_f1 = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(optimizer, scheduler)
            metrics = self.evaluate()
            self.log.emit(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")
            self.log.emit(f"Validation Metrics: {metrics}")
            self.progress.emit(int((epoch + 1) / self.epochs * 100))

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }
                torch.save(checkpoint, Config.SAVE_DIR / 'best_model.pt')
                self.log.emit("Saved best model checkpoint")

        self.finished.emit()

    def train_epoch(self, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            response = batch['response'].to(self.device)
            outputs = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, response)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        preds, targets = [], []
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                response = batch['response'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, response)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                preds.extend(predicted.cpu().numpy())
                targets.extend(response.cpu().numpy())
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': np.mean(np.array(preds) == np.array(targets)),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# GUI Main Window
class AuroraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
        self.model = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Aurora AI Trainer")
        self.setGeometry(100, 100, 800, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Data Section
        data_group = QWidget()
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        data_layout.addWidget(QLabel("1. Data Configuration"))
        self.data_path = QLineEdit("data/train_data.yaml")
        data_layout.addWidget(self.data_path)
        data_btn_layout = QHBoxLayout()
        self.generate_data_btn = QPushButton("Generate Dataset")
        self.generate_data_btn.clicked.connect(self.generate_dataset)
        data_btn_layout.addWidget(self.generate_data_btn)
        self.load_data_btn = QPushButton("Load Existing Data")
        self.load_data_btn.clicked.connect(self.load_data)
        data_btn_layout.addWidget(self.load_data_btn)
        data_layout.addLayout(data_btn_layout)
        self.num_entries = QLineEdit("1000")
        data_layout.addWidget(QLabel("Number of Entries:"))
        data_layout.addWidget(self.num_entries)

        # Model Section
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        model_layout.addWidget(QLabel("2. Model Configuration"))
        self.ollama_model = QComboBox()
        self.ollama_model.addItems(["llama3", "mistral", "custom"])
        model_layout.addWidget(QLabel("Ollama Base Model:"))
        model_layout.addWidget(self.ollama_model)
        self.custom_model = QLineEdit()
        self.custom_model.setEnabled(False)
        self.ollama_model.currentTextChanged.connect(self.toggle_custom_model)
        model_layout.addWidget(QLabel("Custom Model Name (if selected):"))
        model_layout.addWidget(self.custom_model)

        # Training Section
        train_group = QWidget()
        train_layout = QVBoxLayout()
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        train_layout.addWidget(QLabel("3. Training"))
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.train_btn)
        self.progress = QProgressBar()
        train_layout.addWidget(self.progress)

        # Inference Section
        infer_group = QWidget()
        infer_layout = QVBoxLayout()
        infer_group.setLayout(infer_layout)
        layout.addWidget(infer_group)

        infer_layout.addWidget(QLabel("4. Test Inference"))
        self.test_text = QLineEdit("Hello, how can I assist you?")
        infer_layout.addWidget(QLabel("Test Text:"))
        infer_layout.addWidget(self.test_text)
        self.infer_btn = QPushButton("Run Inference")
        self.infer_btn.clicked.connect(self.run_inference)
        infer_layout.addWidget(self.infer_btn)

        # Output Section
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(QLabel("Output Log:"))
        layout.addWidget(self.output)

    def log_message(self, message):
        self.output.append(message)

    def toggle_custom_model(self, text):
        self.custom_model.setEnabled(text == "custom")

    def generate_dataset(self):
        try:
            num_entries = int(self.num_entries.text())
            data_path = self.data_path.text()
            generate_training_data(num_entries=num_entries, output_file=data_path)
            self.log_message(f"Generated dataset with {num_entries} entries at {data_path}")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid number of entries: {e}")

    def load_data(self):
        data_path = QFileDialog.getOpenFileName(self, "Select Data File", str(Config.DATA_DIR), "YAML Files (*.yaml)")[0]
        if data_path:
            self.data_path.setText(data_path)
            self.log_message(f"Loaded data path: {data_path}")

    def load_and_split_data(self):
        data_path = Path(self.data_path.text())
        if not data_path.exists():
            QMessageBox.critical(self, "Error", "Data file not found!")
            return None, None
        
        with open(data_path, 'r') as f:
            data = yaml.safe_load(f)
        
        np.random.seed(Config.SEED)
        indices = np.random.permutation(len(data))
        split = int(len(data) * (1 - Config.VALIDATION_SPLIT))
        train_data = [data[i] for i in indices[:split]]
        val_data = [data[i] for i in indices[split:]]
        
        return train_data, val_data

    def initialize_model(self):
        base_model = self.ollama_model.currentText()
        if base_model == "custom" and self.custom_model.text():
            base_model = self.custom_model.text()
        
        self.log_message(f"Initializing Aurora with Ollama model: {base_model}")
        try:
            self.model = AuroraModel(ollama_model=base_model).to(self.device)
            self.log_message("Model initialized successfully with Ollama weights")
        except Exception as e:
            self.log_message(f"Failed to initialize with Ollama: {e}. Falling back to BERT.")
            self.model = AuroraModel(ollama_model=None).to(self.device)

    def start_training(self):
        if not self.model:
            self.initialize_model()
        
        train_data, val_data = self.load_and_split_data()
        if not train_data or not val_data:
            return
        
        train_dataset = AuroraDataset(train_data, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = AuroraDataset(val_data, self.tokenizer, Config.MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        self.worker = TrainingWorker(self.model, train_loader, val_loader, self.device, Config.EPOCHS)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_message)
        self.worker.finished.connect(lambda: self.log_message("Training completed!"))
        self.worker.start()
        self.log_message("Training started...")

    def run_inference(self):
        if not self.model:
            QMessageBox.critical(self, "Error", "Train or load a model first!")
            return
        
        text = self.test_text.text()
        
        try:
            pred, conf = self.model.predict(text, self.tokenizer, self.device)
            self.log_message(f"Prediction: {pred} (Confidence: {conf:.4f})")
            
            ollama_response = ollama.chat(
                model=self.ollama_model.currentText(),
                messages=[{"role": "user", "content": f"Interpret this prediction: {pred} with confidence {conf} for '{text}'"}]
            )
            self.log_message(f"Ollama Interpretation: {ollama_response['message']['content']}")
        except Exception as e:
            self.log_message(f"Inference failed: {e}")

class AuroraDataset(Dataset):
    def __init__(self, data: list, tokenizer: BertTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = str(item['text'])
        response = torch.tensor(item['response'], dtype=torch.long)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'response': response
        }

    def __len__(self) -> int:
        return len(self.data)

class AuroraModel(nn.Module):
    def __init__(self, ollama_model: str = None, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.use_ollama = ollama_model is not None
        
        if self.use_ollama:
            try:
                # Get embedding size from Ollama model
                test_embedding = ollama.embeddings(model=ollama_model, prompt="test")['embedding']
                self.embedding_dim = len(test_embedding)
                self.embedding_layer = nn.Embedding(30522, self.embedding_dim)  # BERT vocab size
                self.load_ollama_weights(ollama_model)
            except Exception as e:
                logging.warning(f"Ollama loading failed: {e}. Falling back to BERT.")
                self.use_ollama = False
        
        if not self.use_ollama:
            self.embedding_dim = hidden_size
            self.embedding_layer = nn.Embedding.from_pretrained(
                torch.tensor(BertTokenizer.from_pretrained(Config.MODEL_CHECKPOINT).get_vocab().values(), dtype=torch.long).unsqueeze(1),
                freeze=False
            )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=12,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def load_ollama_weights(self, ollama_model: str):
        """Load and adapt Ollama embeddings into the embedding layer"""
        vocab = BertTokenizer.from_pretrained(Config.MODEL_CHECKPOINT).get_vocab()
        with torch.no_grad():
            for token, idx in vocab.items():
                try:
                    embedding = ollama.embeddings(model=ollama_model, prompt=token)['embedding']
                    self.embedding_layer.weight[idx] = torch.tensor(embedding)
                except Exception:
                    continue  # Skip if embedding fetch fails

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(input_ids)
        mask = attention_mask == 0  # Convert to padding mask for Transformer
        encoded = self.encoder(embeddings, src_key_padding_mask=mask)
        pooled = encoded.mean(dim=1)  # Mean pooling over sequence
        return self.classifier(self.dropout(pooled))

    def predict(self, text: str, tokenizer: BertTokenizer, device: torch.device) -> Tuple[int, float]:
        self.eval()
        with torch.no_grad():
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=Config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            return pred, confidence

def main():
    app = QApplication(sys.argv)
    window = AuroraGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()