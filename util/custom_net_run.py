# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.net_run.net_run import TrainInferAgent
from pymic.net_run.net_factory import net_dict
from pymic.util.parse_config import parse_config
from network.unet2dres import UNet2DRes
from network.MGNet import MGNet

local_net_dict = {
    'UNet2DRes': UNet2DRes,
    'MGNet': MGNet
}

def get_network(params):
    net_type = params['net_type']
    if(net_type in local_net_dict):
        return local_net_dict[net_type](params)
    else:
        return net_dict[net_type](params)
    
def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN
    net_param = config['network']
    config['network'] = net_param
    net    = get_network(net_param)

    agent  = TrainInferAgent(config, stage)
    agent.set_network(net)
    agent.run()

if __name__ == "__main__":
    main()
