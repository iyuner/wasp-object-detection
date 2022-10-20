from mmdet.apis import init_detector, inference_detector, show_result_pyplot, \
                        set_random_seed
from mmdet.datasets.cityscapes import CityscapesDataset
import mmcv
import os

demo_images_dir = "data/cityscapes/leftImg8bit/demoVideo/stuttgart_00"
imgs = [os.path.join(demo_images_dir, file) for file in os.listdir(demo_images_dir)]

"""de_detr"""
config_file = 'tutorial_exps/self-de-detr/slurm-4045240/deformable_detr_cityscapes.py'
tested_ckp = "tutorial_exps/self-de-detr/slurm-4045240/epoch_5.pth"    # trained weight
# config_file = 'deformable_detr_r50_16x2_50e_coco.py'
# tested_ckp = "deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"    # pretrained weight

saved_model = init_detector(config_file, tested_ckp, device='cuda')


# show the results
if isinstance(imgs, list):
    for i, img in enumerate(imgs):
        result = inference_detector(saved_model, img)
        show_result_pyplot(
            saved_model,
            img,
            result,
        #     palette=args.palette,
            score_thr=0.3,
            out_file='data/cityscapes/leftImg8bit/demoVideo/processed_stuttgart_00/de_detr_before_train/' + os.path.basename(img)
            )
else:
    result = inference_detector(saved_model, imgs)
    show_result_pyplot(
        saved_model,
        imgs,
        result,
    #     palette=args.palette,
        score_thr=0.3,
        # out_file='cityscpes_de_detr_before_train.png'
        )