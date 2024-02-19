class Path(object):
    @staticmethod
    def db_dir():



            root_dir =['IR_video','RGB_video']
            # Save preprocess data into output_dir
            output_dir = ['IR_frames', 'RGB_frames']
            label_path=['IR_label.txt', 'RGB_label.txt']
            return root_dir, output_dir, label_path


    @staticmethod
    def c3dmodel_dir():
        return 'c3d-pretrained.pth'

    @staticmethod
    def r3dmodel_dir():
        return 'r3d18_K_200ep.pth'

    @staticmethod
    def r2d1dmodel_dir():
        return 'r2p1d18_K_200ep.pth'