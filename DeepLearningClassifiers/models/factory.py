from models.resnet_model import Conv1DResidualClassifier
from models.inceptiontime_model import InceptionTimeClassifier
from models.ffnn_model import FFNNClassifier
from models.cdil_cnn import CONV
from models.lstm_fcn import LSTM_FCN_Univariate
from models.times_net import TimesNetClassifier

import math

def get_model(model_name, input_size, output_size, device, seconds_min=0.49):

    if model_name == "resnet":
        model = Conv1DResidualClassifier(input_channels=input_size, nb_classes=output_size).to(device)  
        print(f"({input_size},{ output_size})")
        #summary(model, input_size=(32, 80, 2))
        return model
    elif model_name == "inception":
        model = InceptionTimeClassifier(input_channels=input_size, nb_classes=output_size).to(device)  
        return model
    elif model_name == "ffnn":
        #input_size = X_train.shape[1] * input_size 
        #input_size = 16 * input_size  # 32 is the batch size
        model = FFNNClassifier(input_size=input_size, output_size=output_size, hidden_size=4096, num_layers=12).to(device)
        #summary(model, input_size=(32, 8000))
        return model
    elif model_name == "conv":
        #model = CONV(task='classification', model='CDIL', input_size=input_size, output_size=output_size, num_channels=[80, 16], kernel_size=3, deformable=True, dynamic=False).to(device)
        T = 12569
        L = math.ceil(math.log2(T))
        base = [32, 64, 128, 256]
        num_channels = base + [256]*(L-len(base))
        return CONV(task='classification', model='CDIL', input_size=input_size, output_size=output_size,
                num_channels=num_channels, kernel_size=3, deformable=True, dynamic=False).to(device)
    elif model_name == "lstm_fcn":
        model = LSTM_FCN_Univariate(input_size=input_size, num_classes=output_size).to(device)
        return model
    elif model_name == "times_net":
        input_size = int(seconds_min * 22_050) #TODO refactor this to change the min input time
        return TimesNetClassifier(seq_len=input_size, nb_classes=output_size).to(device)