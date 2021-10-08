import os
from datetime import datetime

#--------------
#util
#--------------
from utils.color import Colorer

class DirectroyMaker:
    C = Colorer.instance()
    sub_dir_type = ['model','log','config']
    def __init__(self, root, save_model=True,save_log=True,save_config=True):
        self.root = os.path.expanduser(root)
        self.save_model  = save_model
        self.save_log = save_log
        self.save_config = save_config
        
        
    def experiments_dir_maker(self,args):
        
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        now = datetime.now()
        time_idx = '_%s-%s-%s-%s-%s-%s' % (now.year,now.month,now.day,now.hour, now.minute,now.second)
        # --save_dir
        detail_dir = args.data_type + "_" + args.classifier_type + "_PSKD_" + str(args.PSKD) + time_idx
        
        create_dir_list = []
        if self.save_model:
            add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[0])
            new_path = os.path.join(self.root,add_subdir_to_detail_dir)
            create_dir_list.append(new_path)
        if self.save_log:
            add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[1])
            new_path = os.path.join(self.root,add_subdir_to_detail_dir)
            create_dir_list.append(new_path)    
        if self.save_config:
            add_subdir_to_detail_dir = os.path.join(detail_dir,self.sub_dir_type[2])
            new_path = os.path.join(self.root,add_subdir_to_detail_dir)
            create_dir_list.append(new_path)
        
        for path in create_dir_list:
            if not os.path.exists(path):
                os.makedirs(path)
                
        print(self.C.violet2("[Directory] save_dir: {}".format(create_dir_list[0])))
        print(self.C.violet2("[Directory] log_dir: {}".format(create_dir_list[1])))
        print(self.C.violet2("[Directory] config_dir: {}".format(create_dir_list[2])))
        
        return create_dir_list