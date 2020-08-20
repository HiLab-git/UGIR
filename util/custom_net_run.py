# -*- coding: utf-8 -*-
from __future__ import print_function, division

import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.net_run_agent import NetRunAgent
from network.unet2dres import UNet2DRes
from network.MGNet import MGNet

local_net_dict = {
    'UNet2DRes': UNet2DRes,
    'MGNet': MGNet
}

    
def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python custom_net_run.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN
    agent  = NetRunAgent(config, stage)
    agent.set_network_dict(local_net_dict)
    agent.run()

if __name__ == "__main__":
    main()
