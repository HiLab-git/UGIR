# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import scipy 
import sys
import time
import torch 
import numpy as np 
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.util.parse_config import parse_config
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.net_run.infer_func import Inferer
from network.unet2dres import UNet2DRes
from network.MGNet import MGNet

net_dict = {
    'UNet2DRes': UNet2DRes,
    'MGNet': MGNet
}

class CustomSegAgent(SegmentationAgent):
    def __init__(self, config, stage = 'train'):
        super(CustomSegAgent, self).__init__(config, stage)

    def infer(self):
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)
        # load network parameters and set the network as evaluation mode
        checkpoint_name = self.get_checkpoint_name()
        checkpoint = torch.load(checkpoint_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        
        if(self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            if(self.config['testing']['test_time_dropout'] == True):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        print('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)

        infer_cfg = self.config['testing']
        infer_cfg['class_num'] = self.config['network']['class_num']
        infer_obj = Inferer(self.net, infer_cfg)
        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loder:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
                start_time = time.time()
                pred = infer_obj.run(images)
                if(isinstance(pred, (tuple, list))):
                    pred = [item.cpu().numpy() for item in pred]
                else:
                    pred = pred.cpu().numpy()
                data['predict'] = pred
                # inverse transform
                for transform in self.transform_list[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 
                
                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)               
                self.save_ouputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        print("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def save_ouputs(self, data):
        output_dir = self.config['testing']['output_dir']
        ignore_dir = self.config['testing'].get('filename_ignore_dir', True)
        save_prob  = self.config['testing'].get('save_probability', False)
        save_var   = self.config['testing'].get('save_multi_pred_var', False)
        multi_pred_avg = self.config['testing'].get('multi_pred_avg', False)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        if(not os.path.exists(output_dir)):
            os.mkdir(output_dir)

        names, pred = data['names'], data['predict']
        if(isinstance(pred, (tuple, list))):
            prob_list  = [scipy.special.softmax(predi,axis=1) for predi in pred]
            prob_stack = np.asarray(prob_list, np.float32)
            var    = np.var(prob_stack, axis = 0)
            if(multi_pred_avg):
                prob   = np.mean(prob_stack, axis = 0)
            else:
                prob = prob_list[0]
        else:
            prob  = scipy.special.softmax(pred, axis = 1) 
        output = np.asarray(np.argmax(prob,  axis = 1), np.uint8)
        if((label_source is not None) and (label_target is not None)):
            output = convert_label(output, label_source, label_target)
        # save the output and (optionally) probability predictions
        root_dir  = self.config['dataset']['root_dir']
        for i in range(len(names)):
            save_name = names[i].split('/')[-1] if ignore_dir else \
                names[i].replace('/', '_')
            if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                save_name = save_name.replace(filename_replace_source, filename_replace_target)
            print(save_name)
            save_name = "{0:}/{1:}".format(output_dir, save_name)
            save_nd_array_as_image(output[i], save_name, root_dir + '/' + names[i])
            save_name_split = save_name.split('.')

            if('.nii.gz' in save_name):
                save_prefix = '.'.join(save_name_split[:-2])
                save_format = 'nii.gz'
            else:
                save_prefix = '.'.join(save_name_split[:-1])
                save_format = save_name_split[-1]
            
            if(save_prob):
                class_num = prob.shape[1]
                for c in range(0, class_num):
                    temp_prob = prob[i][c]
                    prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                    if(len(temp_prob.shape) == 2):
                        temp_prob = np.asarray(temp_prob * 255, np.uint8)
                    save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/' + names[i])

            if(save_var):
                var = var[i][1]
                var_save_name = "{0:}_var.{1:}".format(save_prefix, save_format)
                save_nd_array_as_image(var, var_save_name, root_dir + '/' + names[0])

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python custom_net_run.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    # use custormized CNN
    agent  = CustomSegAgent(config, stage)
    net_name = config['network']['net_type']
    if(net_name in net_dict):
        net = net_dict[net_name](config['network'])
        agent.set_network(net)
        agent.run()
    else:
        raise ValueError("undefined network {0:}".format(net_name))

if __name__ == "__main__":
    main()
