import torch.nn as nn
import torch
class textRCNN(nn.Module):
    def __init__(self, vab_size, max_len, embed_dim, class_num):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vab_size, embedding_dim=embed_dim)
        self.left_rnn = nn.RNN(input_size=embed_dim, hidden_size=embed_dim*2, num_layers=1, batch_first=True)
        self.right_rnn = nn.RNN(input_size=embed_dim, hidden_size=embed_dim*2, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(embed_dim*5, embed_dim),
                                nn.Tanh(),
                                nn.MaxPool1d(kernel_size=embed_dim))
        self.head = nn.Linear(max_len, class_num)

    def forward(self, x):
        x_embed = self.embed(x)
        x_left, left_h = self.left_rnn(x_embed)
        x_right, right_h = self.right_rnn(x_embed.flip(dims=[1]))
        x_concat = torch.concat([x_left, x_embed, x_right], dim=2)
        x_latent = self.fc(x_concat)
        x_latent = x_latent.reshape(x_latent.shape[0], -1)
        output = self.head(x_latent)
        return output

if __name__ == "__main__":
    import torch
    model = textRCNN(6400, 1000, 192, 14)
    x = torch.randint(low=0,  high=6400, size=(10, 1000))
    print(model(x).shape)