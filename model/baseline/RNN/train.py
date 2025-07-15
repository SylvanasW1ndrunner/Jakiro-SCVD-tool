import pickle
from sched import scheduler

import numpy as np
from fontTools.misc.timeTools import epoch_diff
from pyflakes.checker import counter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from SmartContractDetect.embedding import bertembedding
from SmartContractDetect.model.baseline.CNN.cnn import CNN_Model
from SmartContractDetect.model.baseline.CNNdataloader import NNDataset
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from SmartContractDetect.model.baseline.RNN.rnn import RNN_Model

RE_train_csv = "../../../csvdata/contractData/RE_train_files.csv"
RE_test_csv = "../../../csvdata/contractData/RE_test_files.csv"
IO_train_csv = "../../../csvdata/contractData/IO_train_files.csv"
IO_test_csv = "../../../csvdata/contractData/IO_test_files.csv"
TO_train_csv = "../../../csvdata/contractData/TO_train_files.csv"
TO_test_csv = "../../../csvdata/contractData/TO_test_files.csv"

RE_train_dataset = NNDataset(RE_train_csv)
RE_test_dataset = NNDataset(RE_test_csv)
IO_train_dataset = NNDataset(IO_train_csv)
IO_test_dataset = NNDataset(IO_test_csv)
TO_train_dataset = NNDataset(TO_train_csv)
TO_test_dataset = NNDataset(TO_test_csv)

RE_train_dataloader = torch.utils.data.DataLoader(RE_train_dataset, batch_size=64, shuffle=True)
RE_test_dataloader = torch.utils.data.DataLoader(RE_test_dataset, batch_size=64, shuffle=True)
IO_train_dataloader = torch.utils.data.DataLoader(IO_train_dataset, batch_size=64, shuffle=True)
IO_test_dataloader = torch.utils.data.DataLoader(IO_test_dataset, batch_size=64, shuffle=True)
TO_train_dataloader = torch.utils.data.DataLoader(TO_train_dataset, batch_size=64, shuffle=True)
TO_test_dataloader = torch.utils.data.DataLoader(TO_test_dataset, batch_size=64, shuffle=True)

# with open("../../train_RE.pkl", 'rb') as f:
#     RE_train_dataloader = pickle.load(f)
# with open("../../test_RE.pkl", 'rb') as f:
#     RE_test_dataloader = pickle.load(f)
# with open("../../train_IO.pkl", 'rb') as f:
#     IO_train_dataloader = pickle.load(f)
# with open("../../test_IO.pkl", 'rb') as f:
#     IO_test_dataloader = pickle.load(f)
# with open("../../train_TO.pkl", 'rb') as f:
#     TO_train_dataloader = pickle.load(f)
# with open("../../test_TO.pkl", 'rb') as f:
#     TO_test_dataloader = pickle.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Public\Documents\SmartContractDetect\SmartContractDetect\codebert-base")
model.to(device)


def test(test_dataloader, rnnmodel,vunl):
    checkpoint = torch.load(rnnmodel)
    rnnmodel = RNN_Model(768).to(device)
    rnnmodel.load_state_dict(checkpoint)
    rnnmodel.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_dataloader:
            labels = batch['label'].float().view(-1, 1).to(device)
            sourceembedding, cls_embeddings = bertembedding(batch['text'], model, tokenizer)
            sourceembedding = torch.tensor(sourceembedding, dtype=torch.float32).to(device)
            output = rnnmodel(sourceembedding)
            preds = (output > 0.5).float()  # Convert probabilities to binary predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert lists to numpy arrays for evaluation
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    with open(vunl+ '_sc_' + 'result.txt', 'a') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("-" * 50 + "\n")
    # You can also return the metrics if needed for further processing
    return accuracy, precision, recall, f1

def train(train_dataloader, test_dataloader,vunl):
    num_epochs = 20
    patience = 5
    rnnmodel = RNN_Model(768).to(device)
    optimizer = optim.Adam(rnnmodel.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_loss = float('inf')
    counter = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        with tqdm(total=len(train_dataloader),desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_dataloader:
                labels = batch['label'].float().view(-1, 1).to(device)
                sourceembedding,cls_embeddings = bertembedding(batch['text'],model,tokenizer)
                sourceembedding = torch.tensor(sourceembedding,dtype=torch.float32).to(device)
                cls_embeddings = torch.tensor(cls_embeddings,dtype=torch.float32).to(device)
                print(sourceembedding.shape)
                output = rnnmodel(sourceembedding)
                loss = nn.BCELoss()(output, labels)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            counter = 0
            best_loss = avg_loss
            torch.save(rnnmodel.state_dict(), vunl +'_' +'sc_' +'best_rnnmodel.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step(avg_loss)
    ##test
    test(test_dataloader, vunl+'_'+'sc_'+'best_rnnmodel.pth',vunl)

train(TO_train_dataloader, TO_test_dataloader,'TO')
train(RE_train_dataloader, RE_test_dataloader,'RE')
train(IO_train_dataloader, IO_test_dataloader,'IO')

