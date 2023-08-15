from model import *
from torch import nn
import torch.optim as optim
import argparse
from Log import get_logger
import datetime
from loader import _Loader
from Save import *
from loss import GDLoss
import torch

parser = argparse.ArgumentParser(description='eddy tracking prediction')
parser.add_argument('--epochs', default=150, type=int, help='the train epochs of net work')
parser.add_argument('--lr', default=0.0003, type=float, help='the learning rate of the net work')
parser.add_argument('--number_layer', default=64, type=int, help='the number of the GRU/LSTM layer ')
parser.add_argument('--hidden_size', default=256, type=int, help='the hidden size of GRU or LSTM')
parser.add_argument('--encoder_LSTM_size', default=2, type=int, help='the input size of encoder of LSTM')
parser.add_argument('--eddy_GRU_size', default=2, type=int, help='the input size of EddyGRU')
parser.add_argument('--current_GRU_size', default=100, type=int, help='the input size of CuttentGRU')
parser.add_argument('--decoder_input_size', default=256 + 6, type=int, help='the input size of GRU or LSTM')
parser.add_argument('--output_size', default=2, type=int, help='the predict value number')
parser.add_argument('--train_days', default=15, type=int, help='the days of the train data set')
parser.add_argument('--prediction_days', default=7, type=int, help='the number of prediction days')
parser.add_argument('--batch_size', default=4, type=int, help='the batch size of train loader')
parser.add_argument('--train_name', default='training2.h5', type=str, help='the path/name of the train dataset')

args = parser.parse_args()

logger = get_logger()  # create log file

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    loader = _Loader(args.batch_size, args.prediction_days, file_name=args.train_name, flag='train')
    train_loader = loader.data_loader()

    encoder_model = Encoder(data_length=args.train_days, input_size=args.encoder_LSTM_size, hidden_size=args.hidden_size,
                           num_layers=args.number_layer)

    decoder_model = TraceLSTM(input_size=args.decoder_input_size, hidden_size=args.hidden_size,
                               number_layers=args.number_layer, output_size=args.output_size)

    eddy_GRU = GRU(input_size=args.eddy_GRU_size, hidden_size=args.hidden_size,
                          number_layer=args.number_layer)

    current_GRU = Current(input_size=args.current_GRU_size, hidden_size=args.hidden_size,
                          number_layer=args.number_layer)

    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    eddy_GRU = eddy_GRU.to(device)
    current_GRU = current_GRU.to(device)

    criterion = nn.modules.L1Loss(reduction='sum').to(device)
    criterion2 = GDLoss().to(device)
    optimizer = optim.Adam([{'params': encoder_model.parameters(), 'lr': args.lr},
                            {'params': eddy_GRU.parameters(), 'lr': args.lr},
                            {'params': current_GRU.parameters(), 'lr': args.lr},
                            {'params': decoder_model.parameters(), 'lr': args.lr}])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    all_loss = []
    all_verification_loss = []

    network_result_path = save_para(args)

    encoder_model.train()
    decoder_model.train()
    eddy_GRU.train()
    current_GRU.train()

    # pipeline
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (encoder_data, encoder_label) in enumerate(train_loader):
            h0 = torch.zeros(args.number_layer, encoder_data.shape[0], args.hidden_size)
            c0 = torch.zeros(args.number_layer, encoder_data.shape[0], args.hidden_size)

            site_data = encoder_data[:, :, :2]
            speed = encoder_data[:, :, 2].unsqueeze(-1)
            radius = encoder_data[:, :, 3].unsqueeze(-1)
            speed_radius = torch.cat((speed, radius), dim=-1)
            current_data = encoder_data[:, :, 6:]

            encoder_label = encoder_label.to(device)

            encoder_LSTM_output, encoder_LSTM_hn, encoder_LSTM_cn = encoder_model(site_data.to(device), h0.to(device), c0.to(device))
            GRU_output, eddy_GRU_hn = eddy_GRU(speed_radius.to(device), h0.to(device))
            _, current_GRU_hn = current_GRU(current_data.to(device), h0.to(device))

            loss = torch.zeros(1).to(device)
            dis = torch.zeros(1).to(device)

            previous_hn = encoder_LSTM_hn
            previous_cn = encoder_LSTM_cn
            previous_site = torch.cat((site_data[:, -3:, ...].to(device), encoder_label), dim=1).to(device)
            for j in range(args.prediction_days):
                temp_previous_site = previous_site[:, j:j+3, ...].clone()
                b, d, z = temp_previous_site.shape
                temp_previous_site = temp_previous_site.reshape(b, d*z)
                output, hn, cn = decoder_model(torch.cat((temp_previous_site, encoder_LSTM_output), dim=1), previous_hn,
                                               previous_cn, eddy_GRU_hn, current_GRU_hn)
                temp_label = encoder_label[:, j, :]

                temp_loss = criterion(output, temp_label)
                temp_loss2 = criterion2(output.squeeze(-1), temp_label, device)

                loss += temp_loss + temp_loss2
                dis += temp_loss2

                previous_hn = hn
                previous_cn = cn

            loss = loss / args.prediction_days
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                temp_sate = f'{datetime.datetime.now()}: Index: {i}, Epoch: {epoch}, Loss: {loss.item()}, ' \
                            f'CentralAngle:{dis.item() / args.prediction_days}'
                print(temp_sate)
                logger.info(temp_sate)

        save_state(epoch, network_result_path, encoder_model, decoder_model, eddy_GRU, current_GRU)
        save_loss(epoch, network_result_path, all_loss, all_verification_loss)

        scheduler.step(epoch_loss / len(train_loader))
        all_loss.append(epoch_loss / len(train_loader))

        epoch_state = f'{datetime.datetime.now()}: Epoch: {epoch}, Train Loss: {epoch_loss/len(train_loader)}, Learning rate:{optimizer.param_groups[-1]["lr"]}'
        print(epoch_state)
        logger.info(epoch_state)
        print('-'*60)
