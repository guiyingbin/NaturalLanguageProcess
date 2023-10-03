import torch.nn as nn


class textCNN(nn.Module):
    def __init__(self, vab_size, max_len, embed_dim, class_num):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vab_size, embedding_dim=embed_dim)
        self.cnn = nn.Sequential(*[
            nn.Conv1d(max_len, max_len, kernel_size=3, padding=1),
            nn.BatchNorm1d(max_len),
            nn.ReLU()
        ])
        self.head = nn.Linear(max_len, class_num)
        self.pooling = nn.MaxPool1d(embed_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.cnn(x)
        x = self.pooling(x)
        x = x.reshape(x.shape[0], -1)
        output = self.head(x)
        return output

if __name__ == "__main__":
    import torch
    model = textCNN(6400, 1000, 192, 14)
    x = torch.randint(low=0,  high=6400, size=(10, 1000))
    print(model(x).shape)
