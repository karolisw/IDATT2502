import torch
import torch.nn as nn
import numpy as np


class LongShortTermMemoryModel(nn.Module):


    def __init__(self, encoding_size, label_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, label_size)

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


# Creating a 26 x 26 identiy matrix holding an encoding for each character in the alphabet
char_encodings = np.identity(27) # The 1st is the blank space
char_encoding_size = len(char_encodings)

# All the encodings for use in x_train
ape_encoding = [[char_encodings[1]], [char_encodings[16]], [char_encodings[5]], [char_encodings[0]], [char_encodings[0]]]     # 'ape  '
lion_encoding = [[char_encodings[12]], [char_encodings[9]], [char_encodings[15]], [char_encodings[14]], [char_encodings[0]]]  # 'lion '
horse_encoding = [[char_encodings[8]], [char_encodings[15]], [char_encodings[18]], [char_encodings[19]], [char_encodings[5]]] # 'horse'
camel_encoding = [[char_encodings[3]], [char_encodings[1]], [char_encodings[13]], [char_encodings[5]], [char_encodings[12]]]  # 'camel'
rat_encoding = [[char_encodings[18]], [char_encodings[1]], [char_encodings[20]], [char_encodings[0]], [char_encodings[0]]]    # 'rat  '

# Each identity matrix holds the encoding for a letter of the alphabet (index 0 holds space)
char_indexes = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# A dictionary that holds all the encodings and their individual emoji
emojis = {'ape  ':'🐒', 
          'lion ': '🐕',
          'horse': '🐎', 
          'camel': '🐪', 
          'rat  ': '🐁'}

emojis_encoded = {
    'ape  ': [1., 0., 0., 0., 0.],
    'lion ': [0., 1., 0., 0., 0.],
    'horse': [0., 0., 1., 0., 0.],
    'camel': [0., 0., 0., 1., 0.],
    'rat  ': [0., 0., 0., 0., 1.]
}

emojis_encoding_size = len(emojis)

emoji_indexes = [
                emojis['ape  '], 
                emojis['lion '],
                emojis['horse'], 
                emojis['camel'],
                emojis['rat  ']]


# From index to char array
index_to_char_ape = ['a','p','e',' ', ' ']
index_to_char_lion = ['l', 'i','o','n',' ']
index_to_char_horse= ['h','o','r','s','e']
index_to_char_camel = ['c','a','m','e','l']
index_to_char_rat = ['r','a','t',' ',' ']

# Map each key to a char array
key_encoding_map = {'ape  ':index_to_char_ape, 'lion ': index_to_char_lion, 'horse': index_to_char_horse, 'camel': index_to_char_camel, 'rat  ': index_to_char_rat}

# x_train contains the encodings for all the emojis
x_train = torch.tensor([[ape_encoding], 
                        [lion_encoding], 
                        [horse_encoding], 
                        [camel_encoding], 
                        [rat_encoding]]) 

# y_train contains the keys that each encoding outputs to (retrieved from the dictionary) #TODO could be that y_train should also contain the encodings....
# y_train = torch.tensor([['ape'], ['lion'], ['horse'], ['camel'],['rat']]) 
y_train = torch.tensor([
    [emojis_encoded['ape  '], emojis_encoded['ape  '],emojis_encoded['ape  '],emojis_encoded['ape  '], emojis_encoded['ape  ']],
    [emojis_encoded['lion '], emojis_encoded['lion '], emojis_encoded['lion '], emojis_encoded['lion '], emojis_encoded['lion ']],
    [emojis_encoded['horse'], emojis_encoded['horse'], emojis_encoded['horse'], emojis_encoded['horse'], emojis_encoded['horse']],
    [emojis_encoded['camel'], emojis_encoded['camel'], emojis_encoded['camel'], emojis_encoded['camel'], emojis_encoded['camel']],
    [emojis_encoded['rat  '], emojis_encoded['rat  '], emojis_encoded['rat  '], emojis_encoded['rat  '], emojis_encoded['rat  ']]]) 

model = LongShortTermMemoryModel(char_encoding_size, emojis_encoding_size)

# x train and y train size
print("x train: ", x_train.shape)
print("y train: ", y_train.shape)

x_train = x_train.reshape(5,5,1,27)


optimizer = torch.optim.RMSprop(model.parameters(), 0.001)

for epoch in range(500):
    for i in range(len(x_train)):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

def run(arg: str):
    model.reset()
    y = ''
    for i, c in enumerate(arg):
        y = model.f(torch.tensor([[char_encodings[c]]]))

    print(emoji_indexes[y.argmax()])

while True:
    run(input('Type emoji-name:'))