from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from BLIP.models.blip import blip_decoder

device = torch.device("cpu")


def load_image(image_size, device, image_path):
    raw_image = Image.open(image_path).convert('RGB')   

    # w,h = raw_image.size
    # print(w,h)
    # raw_image.resize((w//5,h//5))
    #display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def get_image_context(image_path, model):

    print(image_path)

    image_size = 384
    image = load_image(image_size=image_size, device=device, image_path=image_path)


    #model_path = './BLIP/models/model_base_caption_capfilt_large.pth'
    # model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
    # model.eval()
    # model = model.to(device)


    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 

        # nucleus sampling
        # caption2 = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        # print('context one: '+caption[0])
        # print('context two: '+caption2[0])
        return caption[0]

