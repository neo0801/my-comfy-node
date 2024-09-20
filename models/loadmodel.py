import torch
from . import model_util
from .pix2pix_model import define_G as pix2pix_G
from .pix2pixHD_model import define_G as pix2pixHD_G
from .BiSeNet_model import BiSeNet
from .BVDNet import define_G as video_G

def show_paramsnumber(net,netname='net'):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    print(netname+' parameters: '+str(parameters)+'M')

def load_pix2pix_model(model_path, netG_name):
    if netG_name == 'HD':
        netG = pix2pixHD_G(3, 3, 64, 'global' ,4)
    else:
        netG = pix2pix_G(3, 3, 64, netG_name, norm='batch',use_dropout=True, init_type='normal', gpu_ids=[])
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(model_path))
    netG = model_util.todevice(netG)
    netG.eval()
    return netG    

def load_video_model(model_path):
    netG = video_G(N=2,n_blocks=4)
    show_paramsnumber(netG,'netG')
    netG.load_state_dict(torch.load(model_path))
    netG = model_util.todevice(netG)
    netG.eval()
    return netG

def bisenet(opt,type='roi'):
    '''
    type: roi or mosaic
    '''
    net = BiSeNet(num_classes=1, context_path='resnet18',train_flag=False)
    show_paramsnumber(net,'segment')
    if type == 'roi':
        net.load_state_dict(torch.load(opt.model_path))
    elif type == 'mosaic':
        net.load_state_dict(torch.load(opt.mosaic_position_model_path))
    net = model_util.todevice(net,opt.gpu_id)
    net.eval()
    return net

def load_mosaic_bisenet(mosaic_position_model_path):
    net = BiSeNet(num_classes=1, context_path='resnet18',train_flag=False)
    show_paramsnumber(net,'segment')
    net.load_state_dict(torch.load(mosaic_position_model_path))
    net = model_util.todevice(net)
    net.eval()
    return net
