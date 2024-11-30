import argparse
import os
import json
import cv2
from ultralytics import YOLO

def process_and_show_video(input_path, model_path, output_path, class_names=None):
    # Проверяем наличие входного файла
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден.")
        return

    # Загружаем модель YOLO
    print("Загрузка модели YOLO...")
    model = YOLO(model_path)

    # Открываем видео с использованием OpenCV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеофайл.")
        return

    # Получаем параметры видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # Исходный кодек

    # Расшифровка кодека FOURCC в строку
    codec = "".join([chr((input_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Исходный кодек видео: {codec}")
    print(f"Частота кадров (FPS): {fps}")
    print(f"Размеры кадра: {width}x{height}")

    # Проверяем путь для выходного файла
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Попробуем использовать исходный кодек, если он поддерживается
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
    except:
        print(f"Ошибка: Кодек {codec} не поддерживается для записи. Используем MP4V.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Резервный кодек

    # Создаём объект для записи видео
    print(f"Сохранение обработанного видео в: {output_path}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Ошибка: Не удалось открыть VideoWriter. Проверьте кодек, путь и параметры.")
        return

    # Обработка видео
    print("Начало обработки видео. Нажмите 'q', чтобы выйти.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Распознавание объектов на кадре
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Преобразование координат в целые числа
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cls = int(box.cls[0])
                confidence = box.conf[0] * 100

                # Получаем имя класса из JSON, если доступно
                class_name = class_names[cls] if class_names and cls < len(class_names) else "Unknown"

                # Добавление текста на кадр
                cv2.putText(frame, f"{class_name} {confidence:.2f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Показываем обработанный кадр в окне
        cv2.imshow('Processed Video', frame)

        # Сохраняем обработанный кадр в видеофайл
        out.write(frame)

        # Прерывание по нажатию клавиши 'q'
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Управляем скоростью воспроизведения
            break

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Обработка завершена.")


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Видеообработка с использованием YOLO.")
    parser.add_argument("input", help="Путь к входному видеофайлу.")
    parser.add_argument("model", help="Путь к файлу модели YOLO.")
    parser.add_argument("output", help="Путь для сохранения выходного видео.")
    parser.add_argument("--classes", help="Путь к JSON-файлу с именами классов.", default=None)

    args = parser.parse_args()

    # Загрузка имён классов из JSON, если указано
    class_names = None
    if args.classes:
        if os.path.exists(args.classes):
            with open(args.classes, "r") as f:
                class_names = json.load(f)
            print(f"Имена классов загружены: {class_names}")
        else:
            print(f"Ошибка: Файл {args.classes} не найден.")

    # Запуск обработки
    process_and_show_video(args.input, args.model, args.output, class_names)


if __name__ == "__main__":
    main()
