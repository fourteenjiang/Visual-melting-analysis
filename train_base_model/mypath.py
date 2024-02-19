class Path(object):
    @staticmethod
    def db_dir(type_video):
        if type_video == 'gray':


            root_dir = 'IR_video'
            # Save preprocess data into output_dir
            output_dir = 'IR_frames'
            label_path='IR_label.txt'
            return root_dir, output_dir, label_path

        elif type_video == 'rgb':

            root_dir = 'RGB_video'
            # Save preprocess data into output_dir
            output_dir = 'RGB_frames'
            label_path = 'RGB_label.txt'
            return root_dir, output_dir, label_path


        else:
            print('Database {} not available.'.format(type_video))
            raise NotImplementedError

    @staticmethod
    def c3dmodel_dir():
        return 'c3d-pretrained.pth'

    @staticmethod
    def r3dmodel_dir():
        return 'r3d18_K_200ep.pth'

    @staticmethod
    def r2d1dmodel_dir():
        return 'r2p1d18_K_200ep.pth'