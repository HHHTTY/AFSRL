Joint data and feature augmentation for self-supervised representation learning on point clouds

> #### Abstract :
> To deal with the exhausting annotations, self-supervised representation learning from unlabeled point clouds has drawn much attention, especially centered on augmentation-based contrastive methods. However, specific augmentations hardly produce sufficient transferability to high-level tasks on different datasets. Besides, augmentations on point clouds may also change underlying semantics. To address the issues, we propose a simple but efficient augmentation fusion contrastive learning framework to combine data augmentations in Euclidean space and feature augmentations in feature space. In particular, we propose a data augmentation method based on sampling and graph generation. Meanwhile, we design a data augmentation network to enable a correspondence of representations by maximizing consistency between augmented graph pairs.
We further design a feature augmentation network that encourages the model to learn representations invariant to the perturbations using an encoder perturbation. We comprehensively conduct extensive object classification experiments and object part segmentation experiments to validate the transferability of the proposed framework. Experimental results demonstrate that the proposed framework is effective to learn the point cloud representation in a self-supervised manner, and yields state-of-the-art results in the community.



## Train AFSRL

train_AFSRL.py




