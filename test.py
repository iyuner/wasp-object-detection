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
config_file = 'configs/deformable_detr/deformable_detr_cityscapes.py'
tested_ckp = "deformable_detr_cityscapes_epoch_8.pth"
# pretrained weight
# config_file = 'deformable_detr_r50_16x2_50e_coco.py'
# tested_ckp = "deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth"    

"""fater rcnn"""
# trained weight
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cityscapes.py'
# tested_ckp = "faster_rcnn_cityscapes_epoch_8.pth"
# pretrained weight
# config_file = 'faster_rcnn_r50_fpn_1x_coco.py'
# tested_ckp = "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"


saved_model = init_detector(config_file, tested_ckp, device='cpu')

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