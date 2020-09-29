from datetime import datetime
from pathlib import Path

from torchvision.utils import save_image


def timestamp() -> str:
    def extract_digits(line: str) -> str:
        digits = ''
        for c in line:
            if c.isdigit():
                digits += c
        return digits

    def datetime_millisec() -> str:
        return datetime.now().isoformat(timespec='seconds')

    return extract_digits(datetime_millisec())

def save_imgs(save_dir, imgs):
    for i, img in enumerate(imgs):
        save_image(img, save_dir / f'{timestamp()}{i}.png')
