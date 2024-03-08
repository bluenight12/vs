import numpy as np
from PIL import Image
import os
import sys
import time
import cv2
from predict_pose import generate_pose_keypoints
import test
from U_2_Net import u2net_load
from U_2_Net import u2net_run


class RUN:

    def __init__(self):
        self.u2net = u2net_load.model(model_name='u2netp')

    #Cloth mask
    def get_cloth_mask(self):
        #u2net = u2net_load.model(model_name = 'u2netp')
        #cloth_name = f'cloth_{int(time.time())}.png'
        self.cloth_name = 'cloth001.png'

        cloth_path = os.path.join(
            'inputs/cloth', sorted(os.listdir('inputs/cloth'))[-1])
        cloth = Image.open(cloth_path)
        cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
        cloth.save(os.path.join('Data_preprocessing/test_color', self.cloth_name))
        os.system('export CUDA_VISIBLE_DEVICES=""')
        u2net_run.infer(self.u2net, 'Data_preprocessing/test_color',
                        'Data_preprocessing/test_edge')

        Image.open(f'Data_preprocessing/test_edge/{self.cloth_name}')

    def get_image(self):
        #img_name = f'img_{int(time.time())}.png'
        self.img_name = 'img001.png'

        img_path = os.path.join('inputs/img', sorted(os.listdir('inputs/img'))[-1])
        img = Image.open(img_path)
        img = img.resize((192, 256), Image.BICUBIC)

        img_path = os.path.join('Data_preprocessing/test_img', self.img_name)
        img.save(img_path)
        #os.system('export CUDA_VISIBLE_DEVICES="0"')
        os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'lip_final.pth' --input-dir 'Data_preprocessing/test_img' --output-dir 'Data_preprocessing/test_label'")

        pose_path = os.path.join('Data_preprocessing/test_pose',
                                self.img_name.replace('.png', '_keypoints.json'))
        generate_pose_keypoints(img_path, pose_path)

    def run_test(self):
        os.system('rm -rf Data_preprocessing/test_pairs.txt')
        with open('Data_preprocessing/test_pairs.txt', 'w') as f:
            f.write(f'{self.img_name} {self.cloth_name}')
        test.main()
        output_grid = np.concatenate([
            np.array(Image.open(f'Data_preprocessing/test_img/{self.img_name}')),
            np.array(Image.open(f'Data_preprocessing/test_color/{self.cloth_name}')),
            np.array(Image.open(f'results/test/try-on/{self.img_name}'))
        ], axis=1)

        image_grid = Image.fromarray(output_grid)
        
        cv2.imshow('image', cv2.cvtColor(output_grid, cv2.COLOR_RGB2BGR))
        if cv2.waitKey() & 0xff == 27:
            cv2.destroyAllWindows()


run = RUN()

run.get_cloth_mask()
run.get_image()
run.run_test()


# u2net = u2net_load.model(model_name = 'u2netp')
#cloth_name = f'cloth_{int(time.time())}.png'
# cloth_name = 'cloth001.png'

# cloth_path = os.path.join(
#     'inputs/cloth', sorted(os.listdir('inputs/cloth'))[-1])
# cloth = Image.open(cloth_path)
# cloth = cloth.resize((192, 256), Image.BICUBIC).convert('RGB')
# cloth.save(os.path.join('Data_preprocessing/test_color', cloth_name))
# os.system('export CUDA_VISIBLE_DEVICES=""')
# u2net_run.infer(u2net, 'Data_preprocessing/test_color',
#                 'Data_preprocessing/test_edge')

# Image.open(f'Data_preprocessing/test_edge/{cloth_name}')

# img_name = f'img_{int(time.time())}.png'
# img_name = 'img001.png'

# img_path = os.path.join('inputs/img', sorted(os.listdir('inputs/img'))[-1])
# img = Image.open(img_path)
# img = img.resize((192, 256), Image.BICUBIC)

# img_path = os.path.join('Data_preprocessing/test_img', img_name)
# img.save(img_path)
# #os.system('export CUDA_VISIBLE_DEVICES="0"')
# os.system("python Self-Correction-Human-Parsing-for-ACGPN/simple_extractor.py --dataset 'lip' --model-restore 'lip_final.pth' --input-dir 'Data_preprocessing/test_img' --output-dir 'Data_preprocessing/test_label'")

# pose_path = os.path.join('Data_preprocessing/test_pose',
#                          img_name.replace('.png', '_keypoints.json'))
# generate_pose_keypoints(img_path, pose_path)

# os.system('rm -rf Data_preprocessing/test_pairs.txt')
# with open('Data_preprocessing/test_pairs.txt', 'w') as f:
#     f.write(f'{img_name} {cloth_name}')
# os.system('python test.py')
# output_grid = np.concatenate([
#     np.array(Image.open(f'Data_preprocessing/test_img/{img_name}')),
#     np.array(Image.open(f'Data_preprocessing/test_color/{cloth_name}')),
#     np.array(Image.open(f'results/test/try-on/{img_name}'))
# ], axis=1)

# image_grid = Image.fromarray(output_grid)

# image_grid