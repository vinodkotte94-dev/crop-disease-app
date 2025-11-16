from pathlib import PosixPath
import pathlib
pathlib.WindowsPath = PosixPath

from fastai.vision.all import *

def main():
    # Step 1: Download sample dataset
    path = untar_data('https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz')

    # Step 2: Create a DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=using_attr(RegexLabeller(r'(.+)_\d+\.jpg$'), 'name'),
        item_tfms=Resize(224)
    )

    # Step 3: DataLoaders with num_workers=0 to avoid Windows pickle errors
    dls = dblock.dataloaders(path/"images", bs=16, num_workers=0)

    # Step 4: Create learner
    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # Step 5: Train for 1 epoch
    learn.fine_tune(1)

    # Step 6: Export the model
    learn.export("tomato_disease_model.pkl")

    print("✅ Model exported as model.pkl")

if __name__ == "__main__":
    main()
