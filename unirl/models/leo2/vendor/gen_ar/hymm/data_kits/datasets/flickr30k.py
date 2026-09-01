import io
import torch
from torch.utils.data import Dataset
from index_kits import ArrowIndexV2
from PIL import Image
from torchvision.transforms import transforms

from processors.arrow_kits import get_table
from hymm.utils.helpers import to_2tuple


class CaptionTestDataset(Dataset):
    def __init__(self, arrow_file, target_size, pad_color=(127, 127, 127), prompt=None, get_item_callback=None,
                 enable_dummy=False):
        super().__init__()
        self.target_size = to_2tuple(target_size)
        self.pad_color = pad_color
        self.data = get_table(arrow_file)

        self.pil_image_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert to tensor and normalize to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )
        self.metric_input_key = "answers"
        self.run_fn_kwargs = {"max_new_tokens": 256}
        if prompt is None:
            self.prompt = "Describe the image briefly."
        else:
            self.prompt = prompt
        if get_item_callback is None:
            self.get_item_callback = lambda x: x
            self.process_image = True
        else:
            self.get_item_callback = get_item_callback
            self.process_image = False

    def __len__(self):
        return len(self.data)

    def get_image(self, index):
        temp = self.data["image"][index].as_py()
        image_bytes = io.BytesIO(temp)
        image_bytes.seek(0)
        pil_image = Image.open(image_bytes).convert("RGB")
        return pil_image

    def __getitem__(self, index):
        is_dummy = index // len(self) > 0
        index = index % len(self)

        image = self.get_image(index)
        if self.process_image:
            image, _ = ArrowIndexV2.resize_and_pad(
                image, self.target_size, resample=Image.Resampling.BICUBIC, pad_color=self.pad_color,
            )
            image = self.pil_image_to_tensor(image)

        ret = self.get_item_callback({
            "ids": self.data["index"][index].as_py(),
            "image": image,
            "prompt": self.prompt,
            "seeds": self.data["seed"][index].as_py(),
            "references": self.data["gt_caption"][index].as_py(),
            "is_dummy": is_dummy,
        })
        return ret


Flickr30kTestDataset = CaptionTestDataset
NoCapsValDataset = CaptionTestDataset
