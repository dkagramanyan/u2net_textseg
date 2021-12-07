import requests
import torch
from torch.autograd import Variable
import numpy as np
# from model import U2NETP
from model import U2NET
from PIL import Image

# model_file = "saved_models/u2net/u2netp.pth"
model_file = "saved_models/u2net/u2net.pth"
# filename = "test.jpg"
filename = "http://s3.aidata.me/ailab/u_2_net_new/tsum.com/botinki/7334578_01_640_square.jpg"
input_size = 320
bg_color = (220, 30, 60)


# Загрузка файла
# Имеем вот такую последовательность:
# pillow -> numpy -> torch


# Грузим картинку с локали или с урла в pillow
def load_image(file_or_url_or_path):
    if isinstance(file_or_url_or_path, str) and file_or_url_or_path.startswith("http"):
        file_or_url_or_path = requests.get(file_or_url_or_path, stream=True).raw
    return Image.open(file_or_url_or_path)


# Конвертиурем pillow в torch
def convert_image(pillow_image):

    image = pillow_image.resize((input_size, input_size))
    image = image.convert('RGB')
    image = np.array(image)
    
    # Конвертируем в LAB
    tmpImg = np.zeros((image.shape[0], image.shape[1],3))
    image = image/np.max(image) # нормализация 
    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229 # Странный конвертор в LAB
    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224 # Зачем? Ведь у scikit есть свой rgb2lab
    tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225 # Но обучали именно на этом
    # Изображение из (320,320,3) Превращает в (3,320,320)
    tmpImg = tmpImg.transpose((2, 0, 1))
    # Превращает (3,320,320) в (1, 3, 320, 320) - именно такой shape нужен модели
    tmpImg = tmpImg[np.newaxis,:,:,:]
    # Возможно если так собрать 10 картинок, (10, 3, 320, 320) обработает все разом
    # Но это не точно, нужно почитать код модели

    # Какая-то обильная конвертация, возможно здесь что-то лишнее
    image = torch.from_numpy(tmpImg)
    image = image.type(torch.FloatTensor)
    image = Variable(image)
    
    return image


# Нормализаиця маски, можно было и в numpy сделать
def normalize(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


# Сохранение результата предикции
# Здесь последовательность следующая:
#   torch -> numpy -> pillow
# Одновременно используется pillow и scikit-image
# можно оставить кого-то одного 
def save_output(image, mask, out_name):
    # Нормализуем (-#,+#) -> (0.0,1.0)
    mask = normalize(mask)
    # У маски выкидываются лишние 'пустые' размерности (320,320,1) -> (320,320)
    mask = mask.squeeze()
    # Конвертируем диапазон (0.0, 1.0) -> (0, 255)
    mask = mask.cpu().data.numpy()*255
    # Перегоняем в pillow
    mask = Image.fromarray(mask).convert("L")
    # Увеличиваем до размера исходного изображения
    mask = mask.resize(image.size, resample=Image.BILINEAR)
    # Оригинал перегоняем в RGBA
    image.convert('RGBA')
    # Заполняем альфа канал
    image.putalpha(mask)
    # Сохраняем
    image.save(out_name)


def main():
    
    # -----------------------------------------------------
    # Загрузка модели медленая, делаем один раз и держим в памяти
    
    # Создаем модель
    # net = U2NETP(3,1)
    net = U2NET(3,1)

    # Загружеам веса
    # В варнингах torch попросил map_location='cpu'
    net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    # иницализация
    net.eval()


    # -----------------------------------------------------
    # Обработка:
    # - грузим картинку
    # - препроцесим
    #   - размер делаем 320х320
    #   - режим в RGB->LAB
    #   - перегоняем в numpy
    # - скрамливаем модели
    # - полученную маску объеденяем с оригинальным изображением
    #   - маску нормализуем
    #   - растягиваем из 320х320 до размера оригинального изображения
    #   - склеиваем

    # Грузим изображение
    pillow_image = load_image(filename)
    torch_image = convert_image(pillow_image)
    
    # Шейп должен быть (1,3,320,320)
    # print(image.shape)

    # обрабаываем изображение
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7 = net(torch_image)

    # Забираем из d1 маску
    # В остальных d# тоже маски но хуже качеством
    mask = d1[:,0,:,:]
    
    # сохраняем результат
    save_output(pillow_image, mask, "test.u2net.png")

    del d1,d2,d3,d4,d5,d6,d7

main()
