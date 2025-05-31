import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2

# 1. Определение констант (должны совпадать с теми, что были при обучении)
CLASS_NAMES = ['benign', 'malignant', 'normal']
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# Словарь для преобразования индекса в имя класса (для интерпретации предсказаний)
idx_to_class = {i: name for i, name in enumerate(CLASS_NAMES)}

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# 2. ВОССОЗДАНИЕ АРХИТЕКТУРЫ МОДЕЛИ
# Это должно быть ТОЧНО ТАКОЕ ЖЕ определение, как при обучении
model_to_load = models.mobilenet_v2(weights=None) # Загружаем без предобученных весов, т.к. свои загрузим

# Изменение первого сверточного слоя для одноканального входа
model_to_load.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

# Изменение последнего слоя (классификатора)
num_ftrs_mobile = model_to_load.classifier[1].in_features
model_to_load.classifier[1] = nn.Linear(num_ftrs_mobile, NUM_CLASSES)

# 3. Загрузка сохраненных весов
pytorch_model_filename = 'Model/model_mobilenet_v2_ultrasound.pth' # Путь к сохраненному файлу

if os.path.exists(pytorch_model_filename):
    # Загружаем веса, указывая map_location, чтобы корректно загрузить
    # модель, сохраненную на GPU, если сейчас используется CPU, и наоборот.
    model_to_load.load_state_dict(torch.load(pytorch_model_filename, map_location=device))
    print(f"Веса модели загружены из {pytorch_model_filename}")

    # Перемещаем модель на выбранное устройство
    model_to_load = model_to_load.to(device)

    # 4. Перевод модели в режим оценки
    model_to_load.eval()
    print("Модель переведена в режим оценки (model.eval())")

else:
    print(f"Файл с весами {pytorch_model_filename} не найден!")
    exit() # Или другая обработка ошибки

# 5. Подготовка входных данных (трансформации для одного изображения)
# Используем те же трансформации, что и для тестовой выборки (test_transforms)
# из ноутбука busi-cnn.ipynb
test_transforms_single = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # Для одноканальных
])

# Пример использования модели для предсказания на новом изображении
def predict_image(image_path, model, transform, device, idx_to_class_map):
    try:
        # Загрузка и преобразование изображения
        # Сначала как OpenCV, чтобы легко конвертировать в grayscale, если нужно
        import cv2
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None

        # Преобразование в grayscale, если оно еще не такое
        if img_cv.ndim == 3 and img_cv.shape[2] == 3:
            gray_img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        elif img_cv.ndim == 2:
            gray_img_cv = img_cv
        else:
            print(f"Неожиданная форма изображения: {img_cv.shape}. Попытка взять первый канал.")
            gray_img_cv = img_cv[:,:,0] if img_cv.ndim == 3 else img_cv

        # Применение трансформаций PyTorch
        # ToPILImage ожидает (H, W, C) или (H, W).
        # Наш gray_img_cv это (H, W) - нормально
        input_tensor = transform(gray_img_cv)
        input_batch = input_tensor.unsqueeze(0) # Создаем мини-батч размером 1
        input_batch = input_batch.to(device)

        with torch.no_grad(): # Отключаем вычисление градиентов
            output = model(input_batch)

        # Получение вероятностей и предсказанного класса
        probabilities = torch.softmax(output, dim=1)[0]
        _, predicted_idx = torch.max(output, 1)
        predicted_label = idx_to_class_map[predicted_idx.item()]

        print(f"\nПредсказание для изображения: {image_path}")
        print(f"Предсказанный класс: {predicted_label} (индекс: {predicted_idx.item()})")
        print("Вероятности по классам:")
        for i, class_name in enumerate(idx_to_class_map.values()):
            print(f"  {class_name}: {probabilities[i].item():.4f}")
        return predicted_label, probabilities

    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")
        return None

# --- Пример вызова предсказания ---
if __name__ == '__main__':
    image_to_predict = "Image_ultrasound_mamma/test_image.png" # Замени на путь к твоему изображению
    if os.path.exists(image_to_predict):
        predict_image(image_to_predict, model_to_load, test_transforms_single, device, idx_to_class)
    else:
        print(f"Тестовое изображение {image_to_predict} не найдено для предсказания.")
