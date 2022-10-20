from mmdet.apis import init_detector, inference_detector, show_result_pyplot, \
                        set_random_seed
from mmdet.datasets.cityscapes import CityscapesDataset
import mmcv


img = "data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_060906_leftImg8bit.png"


# images with rider in validation set
imgs = [
"./data/cityscapes/leftImg8bit/val/munster/munster_000017_000019_leftImg8bit.png",
"./data/cityscapes/leftImg8bit/val/munster/munster_000127_000019_leftImg8bit.png",
"./data/cityscapes/leftImg8bit/val/munster/munster_000078_000019_leftImg8bit.png",
]


"""de_detr"""
# trained weight
config_file = 'tutorial_exps/self-de-detr/slurm-4045240/deformable_detr_cityscapes.py'
tested_ckp = "tutorial_exps/self-de-detr/slurm-4045240/latest.pth"
# pretrained weight
# config_file = 'deformable_detr_r50_16x2_50e_coco.py'
# tested_ckp = "deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"    

saved_model = init_detector(config_file, tested_ckp, device='cpu')



# checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
# model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# test a single image
results = inference_detector(saved_model, imgs)

# show the results
if isinstance(results, list):
    for i, (img, result) in enumerate(zip(imgs, results)):
        show_result_pyplot(
            saved_model,
            img,
            result,
        #     palette=args.palette,
            score_thr=0.3,
            # out_file='cityscpes_de_detr_before_train_' + str(i)+ '.png'
            )
else:
    show_result_pyplot(
        saved_model,
        img,
        results,
    #     palette=args.palette,
        score_thr=0.3,
        # out_file='cityscpes_de_detr_before_train.png'
        )