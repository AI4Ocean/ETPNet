import os
import matplotlib.pyplot as plt
import time
import torch


def save_para(args):
    start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    network_result_path = f'experiment/{start_time}'
    if not os.path.exists(network_result_path):
        os.makedirs(network_result_path)

    argsDict = args.__dict__
    with open(f'{network_result_path}/param.txt', 'w') as f:
        for arg, value in argsDict.items():
            f.writelines(arg + ': ' + str(value) + '\n')
    f.close()

    return network_result_path


def save_loss(epoch, network_result_path, all_loss, all_verification_loss):
    if epoch % 10 == 0:
        train_loss_save_path = f'{network_result_path}/pic/train_loss'
        if not os.path.exists(train_loss_save_path):
            os.makedirs(train_loss_save_path)

        verification_loss_save_path = f'{network_result_path}/pic/verification_loss'
        if not os.path.exists(verification_loss_save_path):
            os.makedirs(verification_loss_save_path)

        with open(f'{network_result_path}/pic/verification_loss/verification_loss_{epoch}.txt', 'w') as f:
            for loss in all_verification_loss:
                f.writelines(str(loss) + '\n')
        f.close()

        with open(f'{network_result_path}/pic/train_loss/train_loss_{epoch}.txt', 'w') as f:
            for loss in all_loss:
                f.writelines(str(loss) + '\n')
        f.close()


def save_state(epoch, network_result_path, encoder_model, decoder_model, eddy_GRU, current_GRU):
        decoder_save_path = f'{network_result_path}/states/decoder'
        encoder_save_path = f'{network_result_path}/states/encoder'
        eddy_GRU_save_path = f'{network_result_path}/states/eddy_GRU'
        current_GRU_save_path = f'{network_result_path}/states/current_GRU'
        if not os.path.exists(decoder_save_path):
            os.makedirs(decoder_save_path)
            os.makedirs(encoder_save_path)
            os.makedirs(eddy_GRU_save_path)
            os.makedirs(current_GRU_save_path)

        if (epoch + 1) % 10 == 0:
            torch.save(encoder_model.state_dict(), f'{encoder_save_path}/{epoch}.pth')
            torch.save(decoder_model.state_dict(), f'{decoder_save_path}/{epoch}.pth')
            torch.save(eddy_GRU.state_dict(), f'{eddy_GRU_save_path}/{epoch}.pth')
            torch.save(current_GRU.state_dict(), f'{current_GRU_save_path}/{epoch}.pth')
