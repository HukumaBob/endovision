import os
import json
import cv2


def init_video_processing(input_path: str, output_path: str):
    """
    Инициализирует объекты VideoCapture и VideoWriter для обработки видео.

    :param input_path: Путь к входному видеофайлу.
    :param output_path: Путь для сохранения обработанного видео.
    :return: Кортеж (VideoCapture, VideoWriter).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {input_path}")
        return None, None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Ошибка: Не удалось создать файл для записи {output_path}")
        cap.release()
        return None, None

    return cap, writer


def process_frame(cap, model, writer, class_names=None):
    """
    Обрабатывает один кадр видео.

    :param cap: Объект VideoCapture для чтения видео.
    :param model: Загруженная YOLO модель.
    :param writer: Объект VideoWriter для записи обработанных кадров.
    :param class_names: Список классов (по желанию).
    :return: Кортеж (обработанный кадр, флаг завершения).
    """
    ret, frame = cap.read()
    if not ret:
        return None, True  # Возвращаем None и флаг завершения

    # Прогоняем кадр через модель
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cls = int(box.cls[0])
            confidence = box.conf[0] * 100
            class_name = class_names[cls] if class_names and cls < len(class_names) else "Unknown"
            cv2.putText(frame, f"{class_name} {confidence:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Записываем обработанный кадр
    writer.write(frame)

    return frame, False


def finalize_processing(cap, writer):
    """
    Освобождает ресурсы, связанные с видеообработкой.

    :param cap: Объект VideoCapture.
    :param writer: Объект VideoWriter.
    """
    if cap:
        cap.release()
    if writer:
        writer.release()
    print("Обработка завершена и ресурсы освобождены.")


def load_classes(classes_path: str):
    """
    Загружает список классов из JSON-файла.

    :param classes_path: Путь к JSON-файлу.
    :return: Список классов или None, если файл не указан.
    """
    if not classes_path or not os.path.exists(classes_path):
        print("JSON с именами классов не указан или не существует.")
        return None

    with open(classes_path, "r") as f:
        return json.load(f)
